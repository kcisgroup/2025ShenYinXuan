import json
import logging
import os
import re
import traceback

import aiohttp
import requests

from common.exception import MyException
from constants.code_enum import (
    DiFyAppEnum,
    DataTypeEnum,
    DiFyCodeEnum,
    SysCodeEnum,
)
from constants.dify_rest_api import DiFyRestApi
from services.db_qadata_process import process
from services.user_service import add_question_record, query_user_qa_record

logger = logging.getLogger(__name__)


class QaContext:
    """问答上下文信息"""

    def __init__(self, token, question, chat_id):
        self.token = token
        self.question = question
        self.chat_id = chat_id


class DiFyRequest:
    """
    DiFy操作服务类
    """

    def __init__(self):
        pass

    async def exec_query(self, res):
        """
        处理来自客户端的请求，向 DiFy 服务发起查询，并将结果流式返回给客户端
        :return:
        """
        try:
            # 获取请求体内容 从res流对象获取request-body
            req_body_content = res.request.body
            # 将字节流解码为字符串
            body_str = req_body_content.decode("utf-8")
            # 将json字符串转换为字典
            req_obj = json.loads(body_str)
            logging.info(f"query param: {body_str}")

            # str(uuid.uuid4())区分不同类型问答
            chat_id = req_obj.get("chat_id")
            qa_type = req_obj.get("qa_type")

            #  使用正则表达式移除所有空白字符（包括空格、制表符、换行符等）
            query = req_obj.get("query")
            cleaned_query = re.sub(r"\s+", "", query)

            # 获取登录用户信息
            token = res.request.headers.get("Authorization")
            if not token:
                raise MyException(SysCodeEnum.c_401)
            if token.startswith("Bearer "):
                token = token.split(" ")[1]

            # 封装问答上下文信息
            qa_context = QaContext(token, cleaned_query, chat_id)

            # 判断请求类别
            app_key = self._get_authorization_token(qa_type)

            # 构建请求参数：请求url、请求体、请求头
            dify_service_url, body_params, headers = self._build_request(cleaned_query, app_key, qa_type)

            # 发送请求并处理响应
            async with aiohttp.ClientSession(read_bufsize=1024 * 16) as session:
                # 创建异步 HTTP 会话，向 DiFy 接口发送 POST 请求
                async with session.post(
                    dify_service_url,
                    headers=headers,
                    json=body_params,
                    timeout=aiohttp.ClientTimeout(total=60 * 2),  # 等待2分钟超时
                ) as response:
                    # 打印响应状态码日志
                    logging.info(f"dify response status: {response.status}")
                    # 响应成功
                    if response.status == 200:
                        # await self.res_begin(res, chat_id)
                        # 当前流式事件的数据类型
                        data_type = ""
                        # 缓存业务数据，用于后续处理
                        bus_data = ""
                        while True:
                            reader = response.content
                            reader._high_water = 10 * 1024 * 1024  # 设置缓冲区大小为10MB，防止内存溢出
                            chunk = await reader.readline()
                            if not chunk:
                                break
                            # 字节流转为字符串
                            str_chunk = chunk.decode("utf-8")
                            # 是否以 data: 开头的 SSE 流式消息
                            if str_chunk.startswith("data"):
                                str_data = str_chunk[5:]
                                # 提取 JSON 数据并解析为 Python 字典
                                data_json = json.loads(str_data)
                                # 提取事件名称、对话 ID、消息 ID、任务 ID
                                event_name = data_json.get("event")
                                conversation_id = data_json.get("conversation_id")
                                message_id = data_json.get("message_id")
                                task_id = data_json.get("task_id")

                                # 如果事件是 message，则提取 answer 字段内容
                                if DiFyCodeEnum.MESSAGE.value[0] == event_name:
                                    answer = data_json.get("answer")
                                    # 若回答以 dify_ 开头，将其拆分为事件列表进行判断
                                    if answer and answer.startswith("dify_"):
                                        event_list = answer.split("_")
                                        if event_list[1] == "0":
                                            # 输出开始，设置当前数据类型为事件列表第 3 项
                                            data_type = event_list[2]
                                            # 如果数据类型是 t02（答案），调用 send_message 发送“开始”消息
                                            if data_type == DataTypeEnum.ANSWER.value[0]:
                                                await self.send_message(
                                                    res,
                                                    qa_context,
                                                    answer,
                                                    {"data": {"messageType": "begin"}, "dataType": data_type},
                                                    qa_type,
                                                    conversation_id,
                                                    message_id,
                                                    task_id,
                                                )

                                        elif event_list[1] == "1":
                                            # 输出结束，设置当前数据类型
                                            data_type = event_list[2]
                                            # 如果数据类型是 t02，发送“结束”消息
                                            if data_type == DataTypeEnum.ANSWER.value[0]:
                                                await self.send_message(
                                                    res,
                                                    qa_context,
                                                    answer,
                                                    {"data": {"messageType": "end"}, "dataType": data_type},
                                                    qa_type,
                                                    conversation_id,
                                                    message_id,
                                                    task_id,
                                                )

                                            # 如果数据类型是 t04（业务数据），调用 process 处理后输出业务数据
                                            elif bus_data and data_type == DataTypeEnum.BUS_DATA.value[0]:
                                                res_data = process(json.loads(bus_data)["data"])
                                                # logging.info(f"chart_data: {res_data}")
                                                await self.send_message(
                                                    res,
                                                    qa_context,
                                                    answer,
                                                    {"data": res_data, "dataType": data_type},
                                                    qa_type,
                                                    conversation_id,
                                                    message_id,
                                                    task_id,
                                                )
                                            # 清空当前数据类型
                                            data_type = ""

                                    # 如果当前处于 t02 类型且非控制指令，则继续发送答案内容
                                    elif len(data_type) > 0:
                                        # 这里输出 t02之间的内容
                                        if data_type == DataTypeEnum.ANSWER.value[0]:
                                            await self.send_message(
                                                res,
                                                qa_context,
                                                answer,
                                                {"data": {"messageType": "continue", "content": answer}, "dataType": data_type},
                                                qa_type,
                                                conversation_id,
                                                message_id,
                                                task_id,
                                            )

                                        # 如果当前处于 t04 类型，缓存业务数据
                                        if data_type == DataTypeEnum.BUS_DATA.value[0]:
                                            bus_data = answer

                                # 如果事件是 error，记录错误信息并发送错误提示到客户端
                                elif DiFyCodeEnum.MESSAGE_ERROR.value[0] == event_name:
                                    # 输出异常情况日志
                                    error_msg = data_json.get("message")
                                    logging.error(f"Error 调用dify失败错误信息: {data_json}")
                                    await res.write(
                                        "data:"
                                        + json.dumps(
                                            {
                                                "data": {"messageType": "error", "content": "调用失败请查看dify日志,错误信息: " + error_msg},
                                                "dataType": DataTypeEnum.ANSWER.value[0],
                                            },
                                            ensure_ascii=False,
                                        )
                                        + "\n\n"
                                    )

        except Exception as e:
            logging.error(f"Error during get_answer: {e}")
            traceback.print_exception(e)
            return {"error": str(e)}  # 返回错误信息作为字典
        finally:
            await self.res_end(res)

    async def handle_think_tag(self, answer):
        """
        处理<think>标签内的内容
        :param answer
        """
        think_content = re.search(r"<think>(.*?)</think>", answer, re.DOTALL).group(1)
        remaining_content = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        return think_content, remaining_content

    async def save_message(self, response, message, qa_context, conversation_id, message_id, task_id, qa_type):
        """
            保存消息记录并发送SSE数据
        :param response:
        :param message:
        :param qa_context:
        :param conversation_id:
        :param message_id:
        :param task_id:
        :param qa_type:
        :return:
        """
        # 保存用户问答记录 1.保存用户问题 2.保存用户答案 t02 和 t04
        if "content" in message["data"]:
            await add_question_record(
                qa_context.token, conversation_id, message_id, task_id, qa_context.chat_id, qa_context.question, message, "", qa_type
            )
        elif message["dataType"] == DataTypeEnum.BUS_DATA.value[0]:
            await add_question_record(
                qa_context.token, conversation_id, message_id, task_id, qa_context.chat_id, qa_context.question, "", message, qa_type
            )
        await response.write("data:" + json.dumps(message, ensure_ascii=False) + "\n\n")

    async def send_message(self, response, qa_context, answer, message, qa_type, conversation_id, message_id, task_id):
        """
        SSE 格式发送数据，每一行以 data: 开头
        """
        if answer.lstrip().startswith("<think>"):
            # 处理deepseek模型思考过程样式
            think_content, remaining_content = await self.handle_think_tag(answer)

            # 发送<think>标签内的内容
            message = {
                "data": {"messageType": "continue", "content": "> " + think_content.replace("\n", "") + "\n\n" + remaining_content},
                "dataType": "t02",
            }
            await self.save_message(response, message, qa_context, conversation_id, message_id, task_id, qa_type)

        else:
            await self.save_message(response, message, qa_context, conversation_id, message_id, task_id, qa_type)

    @staticmethod
    async def res_begin(res, chat_id):
        """

        :param res:
        :param chat_id:
        :return:
        """
        await res.write(
            "data:"
            + json.dumps(
                {
                    "data": {"id": chat_id},
                    "dataType": DataTypeEnum.TASK_ID.value[0],
                }
            )
            + "\n\n"
        )

    @staticmethod
    async def res_end(res):
        """
        :param res:
        :return:
        """
        await res.write(
            "data:"
            + json.dumps(
                {
                    "data": "DONE",
                    "dataType": DataTypeEnum.STREAM_END.value[0],
                }
            )
            + "\n\n"
        )

    @staticmethod
    def _build_request(query, app_key, qa_type):
        """
        构建请求参数
        :param app_key:
        :param query: 用户问题
        :param qa_type: 问答类型
        :return:
        """
        body_params = {
            "query": query,
            "inputs": {"qa_type": qa_type},
            "response_mode": "streaming",
            "user": "abc-123",
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {app_key}",
        }

        dify_service_url = DiFyRestApi.build_url(DiFyRestApi.DIFY_REST_CHAT)
        return dify_service_url, body_params, headers

    @staticmethod
    def _get_authorization_token(qa_type: str):
        """
            根据请求类别获取api/token
            固定走一个dify流
             app-IzudxfuN8uO2bvuCpUHpWhvH master分支默认的数据问答key
            :param qa_type
        :return:
        """
        # 遍历枚举成员并检查第一个元素是否与测试字符串匹配
        for member in DiFyAppEnum:
            if member.value[0] == qa_type:
                return os.getenv("DIFY_DATABASE_QA_API_KEY")
        else:
            raise ValueError(f"问答类型 '{qa_type}' 不支持")


async def query_dify_suggested(chat_id) -> dict:
    """
    发送反馈给指定的消息ID。

    :param chat_id: 消息的唯一标识符。
    :return: 返回服务器响应。
    """
    # 查询对话记录
    qa_record = query_user_qa_record(chat_id)
    url = DiFyRestApi.replace_path_params(DiFyRestApi.DIFY_REST_SUGGESTED, {"message_id": qa_record[0]["message_id"]})
    api_key = os.getenv("DIFY_DATABASE_QA_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    response = requests.get(url + "?user=abc-123", headers=headers)

    # 检查请求是否成功
    if response.status_code == 200:
        logger.info("Feedback successfully sent.")
        return response.json()
    else:
        logger.error(f"Failed to send feedback. Status code: {response.status_code},Response body: {response.text}")
        raise

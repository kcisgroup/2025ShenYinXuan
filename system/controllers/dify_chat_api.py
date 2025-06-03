import logging

from sanic import Blueprint
from sanic.response import ResponseStream

from common.exception import MyException
from common.res_decorator import async_json_resp
from common.token_decorator import check_token
from constants.code_enum import SysCodeEnum
from services.dify_service import DiFyRequest, query_dify_suggested

bp = Blueprint("DiFyApi", url_prefix="/dify")

dify = DiFyRequest()


@bp.post("/get_answer")
@check_token
async def get_answer(req):
    """
    调用diFy画布获取数据流式返回
    :param req:
    :return:
    """

    try:
        response = ResponseStream(dify.exec_query, content_type="text/event-stream")
        return response
    except Exception as e:
        logging.error(f"Error Invoke diFy: {e}")
        raise MyException(SysCodeEnum.c_9999)


@bp.post("/get_dify_suggested", name="get_dify_suggested")
@check_token
@async_json_resp
async def dify_suggested(request):
    """
    dify问题建议
    :param request:
    :return:
    """
    chat_id = request.json.get("chat_id")
    return await query_dify_suggested(chat_id)

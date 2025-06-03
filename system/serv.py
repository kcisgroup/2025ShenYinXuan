from sanic import Sanic
from sanic.response import empty

import controllers
from common.route_utility import autodiscover
from config import serv
from config.load_env import load_env

# 加载配置文件
load_env()

# 创建Sanic应用实例，并命名
app = Sanic("sanic-web")

# 注册 controllers 模块中的路由（递归加载）
autodiscover(
    app,
    controllers,
    recursive=True,
)

# 将根路径 / 绑定到返回空响应的 lambda 函数
app.route("/")(lambda _: empty())


if __name__ == "__main__":
    app.run(host=serv.host, port=serv.port, workers=serv.workers)

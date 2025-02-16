FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 复制当前目录的内容到容器内
COPY . /app

# 安装所需的 Python 依赖
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# 设置启动命令
CMD ["python", "server.py","--address","0.0.0.0","--port","5000"]

# github上没有模型的参数，若要本地运行，需要将模型参数下载到本地，然后在Dockerfile中添加COPY命令
# COPY ./model_pth /app/model_pth



# 或者你把模型参数下过来，放在当前目录，应该就可以了,不用添加COPY命令


# 最后修改下面这个CMD命令，将你的模型参数的路径传入，上面那个就不用了，注释掉
# CMD ["python", "server.py","--address","0.0.0.0","--port","5000"，"--hammer_checkpoint","你的hammer模型的checkpoint","--mdfend_checkpoint","你的mdfend模型的checkpoint"]
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV HF_ENDPOINT="https://hf-mirror.com"

# 设置工作目录
WORKDIR /app

#ARG CACHE_BUST=1
#ARG LLM_API_KEY="node"
#ARG LLM_URL="none"

# 复制当前目录的内容到容器内
COPY . /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx

RUN  apt-get install -y \
    libglib2.0-0

# 安装所需的 Python 依赖

# RUN pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/

RUN pip install -r requirements.txt

RUN python pre_load.py


# 设置启动命令
# CMD ["python", "server.py","--address","0.0.0.0","--port","5000","--llm_api_key","${LLM_API_KEY}","--llm_url","${LLM_URL}"]

ENTRYPOINT ["python","server.py","--address","0.0.0.0","--port","5000"]

# github上没有模型的参数，若要本地运行，需要将模型参数下载到本地，然后在Dockerfile中添加COPY命令
# COPY ./model_pth /app/model_pth



# 或者你把模型参数下过来，放在当前目录，应该就可以了,不用添加COPY命令


# 最后修改下面这个CMD命令，将你的模型参数的路径传入，上面那个就不用了，注释掉
# CMD ["python", "server.py","--address","0.0.0.0","--port","5000"，"--hammer_checkpoint","你的hammer模型的checkpoint","--mdfend_checkpoint","你的mdfend模型的checkpoint"]

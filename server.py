import argparse

import connexion
import os
import yaml
from flask import jsonify, request

from api import FakeNewsDetector, LLModel

UPLOAD_FOLDER = './image'  # 上传目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

lm = None

deepseekLm = None

content = "现在你要求扮演一位文本助手，要根据以下内容去编写说明文档。先你要扮演一位新闻专家，你要求根据以下材料说明该新闻为何为假或，特别的，我们提供的理由必然准确，同时个别新闻会有图片，我们会提供图片的造假情况，请你结合存在的图片造假情况和文本情况，去说明该新闻为假的理由，请注意，我们提供的材料必然为真实材料，不要质疑材料准确性，假新闻类型以空格为分割，新闻造假类型以假新闻类型为准，若不存在假新闻类型则为文本造假，可能存在的假词以空格为分割，每个词为独立的词，可能存在的假词仅为该token可能是否造假，请结合文本判断，图片可能无法提供，请你以图片存在的情况下，对文本和材料进行分析"


def process_data(data,news_content):
    news_true_or_false = "真" if data["pred_cls"] != 0 else "假"
    news_type = ""
    for type in data["pred_all_multicls"]:
        news_type += type + " "
    news_fake_token = ""
    for word in data["word_preds"]:
        if word["pred"] == 1:
            news_fake_token += word["token"] + " "
    message = f"这是一条{news_true_or_false}新闻，新闻造假类型是{news_type}，新闻内容是{news_content}，新闻中的假词有{news_fake_token}"
    return message


def upload():
    # 获取上传的文件
    file = request.files['file']

    # 检查文件是否存在
    if file:
        # 保存文件
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        return jsonify(message="Image uploaded successfully", image_path=file_path)
    else:
        return jsonify(message="No file uploaded"), 400


def predict(predict_request):
    text = predict_request.get('text')
    image_path = predict_request.get('image')
    mode = predict_request.get('mode')
    pred_cls, pred_all_multicls, box, word_preds = lm.predict(text, image_path, mode)

    res = {
        'pred_cls': int(pred_cls),
        'pred_all_multicls': pred_all_multicls,
        'box': box,
        'word_preds': word_preds
    }

    return jsonify(res)


def llm_analyze(analyse_request):
    data = analyse_request.get('data')
    news_content=analyse_request.get('text')
    analyse = deepseekLm.chat(process_data(data,news_content))
    res = {
        'analyse': analyse
    }
    return jsonify(res)


app = connexion.App(__name__)
app.add_api('server.yaml')

parser = argparse.ArgumentParser()

# 基本配置
parser.add_argument('--config', default='configs/HAMMER.yaml', type=str)
parser.add_argument('--hammer_checkpoint', default='save/best.pth', type=str)
parser.add_argument('--mdfend_checkpoint', default='save/MDFEND-2025-02-14-10_57_02.pth', type=str)
parser.add_argument('--output_dir', default='results', type=str)
parser.add_argument('--device', default='cuda')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--text_encoder', default='bert-base-uncased')

# 分布式训练相关
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--dist-url', default='tcp://127.0.0.1:10031')
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')

# 日志相关
parser.add_argument('--log_num', '-l', type=str)
parser.add_argument('--model_save_epoch', type=int, default=5)
parser.add_argument('--token_momentum', default=False, action='store_true')
parser.add_argument('--test_epoch', default='best', type=str)
parser.add_argument('--log', action='store_true')

# 服务器相关
parser.add_argument("--nodebug", default=True)
parser.add_argument("--address",
                    default="127.0.0.1")  # 0.0.0.0 for nonlocal use
parser.add_argument("--port", default="5001")
parser.add_argument("--nocache", default=False)

parser.add_argument("--no_cors", action='store_true')
if __name__ == '__main__':
    args = parser.parse_args()
    app.run(port=int(args.port), host=args.address)
else:
    args, _ = parser.parse_known_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    # load_projects(args.dir)
    lm = FakeNewsDetector(args=args, config=config)
    deepseekLm = LLModel(url="https://ark.cn-beijing.volces.com/api/v3",
                         api_key='05fe7d3b-aa40-4a54-b080-ac06c5a371b5',
                         system_content=content)

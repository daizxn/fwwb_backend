import argparse

import connexion
import os
import yaml
from flask import send_from_directory, redirect, jsonify, request


from api import FakeNewsDetector

UPLOAD_FOLDER = './image'  # 上传目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



lm = None

def upload():
    # 获取上传的文件
    file = request.files['file']

    # 检查文件是否存在
    if file:
        # 保存文件
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        return jsonify(message="Image uploaded successfully",image_path=file_path)
    else:
        return jsonify(message="No file uploaded"), 400

def analyze(analyze_request):
    text=analyze_request.get('text')
    image_path=analyze_request.get('image')

    pred_cls, pred_all_multicls, box, word_preds= lm.predict(text,image_path)

    res = {
        'pred_cls': int(pred_cls),
        'pred_all_multicls':  pred_all_multicls,
        'box': box,
        'word_preds': word_preds
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

#服务器相关
parser.add_argument("--nodebug", default=True)
parser.add_argument("--address",
                    default="127.0.0.1")  # 0.0.0.0 for nonlocal use
parser.add_argument("--port", default="5001")
parser.add_argument("--nocache", default=False)


parser.add_argument("--no_cors", action='store_true')
if __name__ == '__main__':
    args = parser.parse_args()
    app.run(port=int(args.port),  host=args.address)
else:
    args, _ = parser.parse_known_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    # load_projects(args.dir)
    lm = FakeNewsDetector(args=args,config=config)
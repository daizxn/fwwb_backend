import argparse
from types import SimpleNamespace

import numpy
import numpy as np
import yaml

from transformers import BertTokenizerFast

from models import box_ops
from models.vit import interpolate_pos_embed
from models.HAMMER import HAMMER
import torch
import torch.nn.functional as F
from PIL import Image

from tools.utils import load_and_preprocess_image


def parse_args():
    parser = argparse.ArgumentParser()

    # 基本配置
    parser.add_argument('--config', default='configs/HAMMER.yaml', type=str)
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

    args = parser.parse_args()

    # 加载yaml配置
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # 将yaml配置添加到args
    for k, v in config.items():
        setattr(args, k, v)

    return args, config


class AbstractFakeNewsDetector:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, text, image=None, mode='multi-label'):
        raise NotImplementedError


class FakeNewsDetector(AbstractFakeNewsDetector):
    def __init__(self, args=None, config=None):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)
        self.model = HAMMER(
            args=args,
            config=config,
            text_encoder='bert-base-uncased',
            tokenizer=self.tokenizer,
            init_deit=True)
        checkpoint_dir = f'save/best.pth'
        checkpoint = torch.load(checkpoint_dir, map_location=self.device)
        state_dict = checkpoint['model']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], self.model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        # model.load_state_dict(state_dict)

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, image_path=None, mode='multi-label'):
        text_input = self.tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
        text_input = {k: v.to(self.device) for k, v in text_input.items()}
        text_input = SimpleNamespace(**text_input)
        image = load_and_preprocess_image(image_path).to(self.device)
        with torch.no_grad():
            logits_real_fake, logits_multicls, output_coord, logits_tok, attn_weights = self.model(image=image,
                                                                                                   text=text_input,
                                                                                                   is_train=False,
                                                                                                   return_attention=True,
                                                                                                   mode=mode)

        pred_cls = logits_real_fake.argmax(1).cpu().numpy().item()
        pred_all_multicls = []
        for cls_idx in range(logits_multicls.shape[1]):
            cls_pred = logits_multicls[:, cls_idx]
            cls_pred[cls_pred >= 0] = 1
            cls_pred[cls_pred < 0] = 0
            pred_all_multicls.append(cls_pred.cpu().numpy().item())

        x_c, y_c, w, h = output_coord.unbind(-1)
        b = [x_c, y_c, w, h]
        box = [i.cpu().numpy().item() for i in b]

        logits_tok_reshape = logits_tok.view(-1, 2)
        tok_pred = logits_tok_reshape.argmax(1).cpu().numpy()

        valid_length = text_input.attention_mask[0].sum().item() - 1  # 减去CLS token
        attn_weights = attn_weights[0][:valid_length, :valid_length].cpu().numpy()
        tok_pred = tok_pred[:valid_length]

        word_preds = {}
        for i, token in enumerate(self.tokenizer.convert_ids_to_tokens(text_input.input_ids[0])):
            if i>=valid_length:
                break
            word_preds[token] = tok_pred[i]

        return pred_cls, pred_all_multicls, box, word_preds

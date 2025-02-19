import time
from types import SimpleNamespace

from transformers import BertTokenizerFast


from models.HAMMER import HAMMER
import torch


from tools.utils import load_and_preprocess_image


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
        checkpoint_dir = args.hammer_checkpoint
        checkpoint = torch.load(checkpoint_dir, map_location=self.device)
        self.all_multicls_dict = {0:'face_swap',1:'face_attribute',2:'text_swap',3:'text_attribute'}
        # model.load_state_dict(state_dict)

        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, image_path=None, mode='multi-label'):
        time_start = time.time()
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        with torch.no_grad(), torch.cuda.amp.autocast():
            # Stream 1: 处理文本输入
            with torch.cuda.stream(stream1):
                text_input = self.tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
                text_input = {k: v.to(self.device) for k, v in text_input.items()}
                text_input = SimpleNamespace(**text_input)

            # Stream 2: 处理图像输入
            with torch.cuda.stream(stream2):
                image = load_and_preprocess_image(image_path).to(self.device)

            # 同步两个流，确保输入处理完成
            torch.cuda.synchronize()
        time_end= time.time()
        print('预处理时间：',time_end-time_start)
        # 进行推理
        time_start = time.time()
        with torch.no_grad():
            logits_real_fake, logits_multicls, output_coord, logits_tok, attn_weights = self.model(image=image,
                                                                                                   text=text_input,
                                                                                                   is_train=False,
                                                                                                   return_attention=True,
                                                                                                   mode=mode)

        time_end = time.time()
        print('推理时间：', time_end - time_start)

        time_start = time.time()
        # 并行处理预测结果
        stream1.wait_stream(torch.cuda.current_stream())
        stream2.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream1):
            # 处理分类结果
            pred_cls = logits_real_fake.argmax(1).cpu()
            pred_all_multicls = []
            for cls_idx in range(logits_multicls.shape[1]):
                cls_pred = logits_multicls[:, cls_idx]
                cls_pred[cls_pred >= 0] = 1
                cls_pred[cls_pred < 0] = 0
                if cls_pred == 1:
                    pred_all_multicls.append(self.all_multicls_dict[cls_idx])

        with torch.cuda.stream(stream2):
            # 处理边界框和token预测
            x_c, y_c, w, h = output_coord.unbind(-1)
            box = [i.cpu() for i in [x_c, y_c, w, h]]

            logits_tok_reshape = logits_tok.view(-1, 2)
            tok_pred = logits_tok_reshape.argmax(1).cpu()

            valid_length = text_input.attention_mask[0].sum().item() - 1
            attn_weights = attn_weights[0][:valid_length, :valid_length].cpu()
            tok_pred = tok_pred[:valid_length]

            word_preds = {}
            for i, token in enumerate(self.tokenizer.convert_ids_to_tokens(text_input.input_ids[0])):
                if i >= valid_length:
                    break
                word_preds[token] = int(tok_pred[i])

        # 同步所有流
        torch.cuda.synchronize()

        # 转换为numpy数组并返回结果
        pred_cls = pred_cls.numpy().item()
        box = [i.numpy().item() for i in box]
        time_end = time.time()
        print('后处理时间：', time_end - time_start)

        return pred_cls, pred_all_multicls, box, word_preds

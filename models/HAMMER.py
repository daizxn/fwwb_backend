from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM, BertForTokenClassification

import torch
import torch.nn.functional as F
from torch import nn
from models import box_ops
from tools.multilabel_metrics import get_multi_label
from timm.models.layers import trunc_normal_




from models.mdfend import MDFEND

class HAMMER(nn.Module):
    def __init__(self, 
                 args = None, 
                 config = None,               
                 text_encoder = None,
                 tokenizer = None,
                 init_deit = True
                 ):
        super().__init__()
        
        self.args = args
        self.tokenizer = tokenizer 
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertForTokenClassification.from_pretrained(text_encoder, 
                                                                    config=bert_config, 
                                                                    label_smoothing=config['label_smoothing'],
                                                                       )

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  

        # creat itm head
        self.itm_head = self.build_mlp(input_dim=text_width, output_dim=2)

        # creat bbox head
        self.bbox_head = self.build_mlp(input_dim=text_width, output_dim=4)

        # creat multi-cls head
        self.cls_head = self.build_mlp(input_dim=text_width, output_dim=4)

        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForTokenClassification.from_pretrained(text_encoder, 
                                                                    config=bert_config,
                                                                    label_smoothing=config['label_smoothing'])       
        self.text_proj_m = nn.Linear(text_width, embed_dim)    
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.norm_layer_aggr =nn.LayerNorm(text_width)
        self.cls_token_local = nn.Parameter(torch.zeros(1, 1, text_width))
        self.aggregator = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)

        self.norm_layer_it_cross_atten =nn.LayerNorm(text_width)
        self.it_cross_attn = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)

        trunc_normal_(self.cls_token_local, std=.02)
        self.apply(self._init_weights)


        # add mdfend_model
        self.mdfend_model = MDFEND('bert-base-uncased',domain_num=9)
        #mdfend_model load checkpoint
        self.load_mdfend_checkpoint()
        self.mdfend_model.eval()
        self.share_to_text = nn.Linear(320,768)

        # 添加融合层（如果使用 concat 方式）
        self.fusion_layer = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.LayerNorm(768),
            nn.GELU()
        )

        # 添加可学习的融合权重
        # self.fusion_weight = nn.Parameter(torch.FloatTensor([0.5]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )

    # 在 HAMMER.py 中添加或修改
    def load_mdfend_checkpoint(self):
        # 加载 checkpoint
        print("### 加载 MDFEND 模型权重")
        checkpoint = torch.load(self.args.mdfend_checkpoint, map_location=self.args.device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # 创建新的 state_dict
        new_state_dict = {}
        
        # 修改键名
        for key, value in state_dict.items():
            if key.startswith('bert'):
                # 添加 'model_mdfend.' 前缀
                new_key = f'mdfend_model.{key}'
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # 加载修改后的权重
        self.mdfend_model.load_state_dict(new_state_dict, strict=False)

    def get_bbox_loss(self, output_coord, target_bbox, is_image=None):
        """
        Bounding Box Loss: L1 & GIoU

        Args:
            image_embeds: encoding full images
        """
        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4

        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            # early check of degenerated boxes
            print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz
            loss_giou = 1 - box_ops.generalized_box_iou(boxes1, boxes2)  # bsz

        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes

    def forward(self, image=None, label=None, text=None, fake_image_box=None, fake_text_pos=None, alpha=0, is_train=True, return_attention=False, mode='multi-label',return_logits=False):
        mdfend_input = {}
        mdfend_text_input = {}
        if mode == 'text':


            text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,return_dict = True, mode = 'text')
            sequence_output = text_output[0]

            mdfend_input['domain'] = torch.ones(text.input_ids.size(0), dtype=torch.long).to(text.input_ids.device)
            mdfend_text_input['token_id']=text.input_ids
            mdfend_text_input['mask']=text.attention_mask
            mdfend_input['text']=mdfend_text_input
            if return_logits:
                return self.mdfend_model.forward(mdfend_text_input['token_id'], mdfend_text_input['mask'], mdfend_input['domain'])

            mdfend_label=self.mdfend_model.predict(mdfend_input)

            sequence_output = self.text_encoder.dropout(sequence_output[:,1:]) # [:,1:] for ingoring class token
            logits_tok = self.text_encoder.classifier(sequence_output)



            return mdfend_label,logits_tok

        else:
            image_embeds = self.visual_encoder(image) 
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

            text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                           return_dict = True, mode = 'text')            
            text_embeds = text_output.last_hidden_state
            #forward the positve image-text pair
            # share_feature=self.mdfend_model(text.input_ids, 
            #                                 text.attention_mask, 
            #                                 domain=torch.ones(text.input_ids.size(0), dtype=torch.long).to(text.input_ids.device))  # 设置 domain 为 1)
            # self.share_to_text = nn.Linear(320, 65536 * 768).to(text.input_ids.device)
            # share_embedding = self.share_to_text(share_feature)

            mdfend_feature = self.mdfend_model(
                                    token_id=text.input_ids,
                                    mask=text.attention_mask,
                                    domain=torch.ones(text.input_ids.size(0), dtype=torch.long).to(text.input_ids.device)
            )  # [batch_size, 320]
            mdfend_feature=self.share_to_text(mdfend_feature)
            # if not hasattr(self, 'share_to_text') or self.share_to_text.out_features != seq_length * hidden_size:
            #     self.share_to_text = nn.Linear(320, seq_length * hidden_size).to(text.input_ids.device)
            #
            # mdfend_transformed = self.share_to_text(mdfend_feature)
            # mdfend_transformed = mdfend_transformed.view(batch_size, seq_length, hidden_size)
            #
            # #text_embeds = text_embeds + mdfend_transformed
            # text_embeds=mdfend_transformed

            # 4. 特征融合（三种方式）
            # a. 加权融合
            # fusion_weight = nn.Parameter(torch.FloatTensor([0.5])).to(text.input_ids.device)
            # combined_embeds = fusion_weight * text_embeds + (1 - fusion_weight) * mdfend_transformed

            # b. 注意力融合
            # attention_weights = torch.matmul(text_embeds, mdfend_transformed.transpose(-2, -1))
            # attention_weights = F.softmax(attention_weights, dim=-1)
            # combined_embeds = torch.matmul(attention_weights, mdfend_transformed)

            # c. concat 后接线性层
            # concat_embeds = torch.cat([text_embeds, mdfend_transformed], dim=-1)
            # combined_embeds = self.fusion_layer(concat_embeds)  # 需要添加 fusion_layer 属性
            # text_embeds=combined_embeds
            output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            mode = 'fusion',
                                        )
            output_feature = output_pos.last_hidden_state[:,0,:]
            # combined_feature = torch.cat([output_feature, mdfend_feature], dim=1)
            # fused_feature = self.fusion_layer(combined_feature)
            fusion_weight = nn.Parameter(torch.FloatTensor([0.5])).to(text.input_ids.device)
            fused_feature=fusion_weight*output_feature+(1-fusion_weight)*mdfend_feature
            ##================= IMG ========================## 
            bs = image.size(0)
            cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)

            text_attention_mask_clone = text.attention_mask.clone() # [:,1:] for ingoring class token
            local_feat_padding_mask_text = text_attention_mask_clone==0 # 0 = pad token

            attn_output, attn_weights = self.it_cross_attn(
                query=self.norm_layer_it_cross_atten(image_embeds), 
                key=self.norm_layer_it_cross_atten(text_embeds), 
                value=self.norm_layer_it_cross_atten(text_embeds),
                key_padding_mask=local_feat_padding_mask_text,
                need_weights=True,  # 设置为 True 以获取注意力权重

            )

            local_feat_it_cross_attn = image_embeds + attn_output

            local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local), 
                                              key=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]), 
                                              value=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]))[0]
            output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
            ##================= BIC ========================## 
            logits_real_fake = self.itm_head(fused_feature)
            ##================= MLC ========================## 
            logits_multicls = self.cls_head(output_feature)
            ##================= TMG ========================##   
            input_ids = text.input_ids.clone()
            logits_tok = self.text_encoder(input_ids, 
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        return_logits = True,   
                                        )     
            if return_attention:
                return logits_real_fake, logits_multicls, output_coord, logits_tok, attn_weights
            else:
                return logits_real_fake, logits_multicls, output_coord, logits_tok   


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
        
        
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



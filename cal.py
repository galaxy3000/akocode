import torch
import torch.nn as nn
from transformers import BertTokenizer
from model.Model import Model
# from FakeSV.code.models.SVFEND import SVFENDModel
from thop import profile, clever_format
from utils.tools import *

def get_dummy_input(device,dataset):
    if dataset == 'fakesv':
        tokenizer = pretrain_bert_wwm_token()
    else:
        tokenizer = pretrain_bert_uncased_token()
    text = "This is a dummy sentence for FLOPs testing."
    encoding = tokenizer(text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    
    input_dict = {
        'title_inputid': encoding['input_ids'].to(device),
        'title_mask': encoding['attention_mask'].to(device),
        'audio_feas': torch.randn(1, 36, 1024).to(device),
        'audiofeas_masks': torch.ones(1, 36).long().to(device),
        'frames': torch.randn(1, 30, 1024).to(device),
        'frames_masks': torch.ones(1, 30).long().to(device),
    }
    return input_dict


class ModelWrapper(nn.Module):
    def __init__(self, model, input_dict):
        super().__init__()
        self.model = model
        self.input_dict = input_dict

    def forward(self, dummy_input):
        output = self.model(**self.input_dict)
        return output


def calculate_flops_and_params(dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dict = get_dummy_input(device,dataset)

    model_path = ''
    model = Model(dataset=dataset, model_path=model_path, module_deep=1, n_tokens=16, 
                               num_heads=8, trans_dim=512, dropout=0.1)
    model.eval()

    wrapper = ModelWrapper(model, input_dict).to(device)
    dummy_input = torch.zeros(1).to(device)

    # 用 thop 计算总 FLOPs
    flops, _ = profile(wrapper, inputs=(dummy_input,), verbose=False)
    print(_/1000000)
    # 过滤 trainable 参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops, trainable_params = clever_format([flops, trainable_params], "%.3f")

    print(f" FLOPs: {flops}")
    print(f" Parameters: {trainable_params}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型总参数量：{total / 1e6:.2f} M")
    print(f"可训练参数量：{trainable / 1e6:.2f} M")



if __name__ == "__main__":
    calculate_flops_and_params('fakett')

    
    
    
#Fakingreceip

# from FakingRecipe.model.FakingRecipe import FakingRecipe_Model


# input_data = {
#     'all_phrase_semantic_fea': torch.randn(1, 33, 512).cuda(),  # fakett -> semantic dim = 512
#     'all_phrase_emo_fea': torch.randn(1, 768).cuda(),
#     'raw_visual_frames': torch.randn(1, 83, 512).cuda(),
#     'raw_audio_emo': torch.randn(1, 768).cuda(),

#     'ocr_phrase_fea': [torch.randn(80, 512).cuda()],
#     'ocr_time_region': [[(0, 10), (15, 30), (40, 50)]],
#     'visual_frames_fea': torch.randn(1, 83, 512).cuda(),
#     'visual_frames_seg_indicator': [torch.randint(0, 5, (83,)).cuda()],
#     'visual_seg_paded': [[(0,10), (11,25), (26,40), (41,55), (56,70)]],
#     'fps': [25.0],
#     'total_frame': [83],
#     'ocr_pattern_fea': torch.randn(1, 256, 64, 64).cuda()
# }


# class WrapperModel(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#     def forward(self, x):
#         return self.model(**x)[0]  # 只返回主输出即可


# model = FakingRecipe_Model(dataset='fakett').cuda()
# model.eval()
# wrapper = WrapperModel(model)

# # 使用 thop 计算 FLOPs 和 参数量
# with torch.no_grad():
#     flops, params = profile(wrapper, inputs=(input_data,), verbose=False)
#     flops, params = clever_format([flops, params], "%.3f")
#     print("FLOPs:",flops)
#     print("Params:",params)




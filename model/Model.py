from utils.tools import *
import torch
from torch import nn
from model.CMA import CrossModalAlignment
from model.CMFR import CMFR
from model.UnifiedTransformer import UnifiedTransformer
import torch.nn.functional as F    

class AttnPooling(nn.Module):
    def __init__(self, feat_dim=512, compressed_dim=128):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, feat_dim))  
        
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim, 
            num_heads=4,
            kdim=feat_dim,
            vdim=feat_dim
        )
        
        self.proj = nn.Linear(feat_dim, compressed_dim)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.query.expand(B, -1, -1)  # [B, 1, 512]
        
        attn_output, _ = self.attn(
            query=q.transpose(0, 1),  # [1, B, 512]
            key=x.transpose(0, 1),    # [L, B, 512]
            value=x.transpose(0, 1)   # [L, B, 512]
        )
        
        return self.proj(attn_output.squeeze(0))
    
    
class Model(torch.nn.Module):
    def __init__(self, dataset, model_path, module_deep, n_tokens, num_heads, trans_dim, dropout):
        super(Model, self).__init__()
        self.dataset = dataset
        self.dropout = dropout
        self.save_param_dir = model_path
        self.n_tokens = n_tokens
        self.num_heads = num_heads
        self.trans_dim = trans_dim
        
        if dataset == 'fakesv':
            self.bert = pretrain_bert_wwm_model()
            self.text_dim = 1024
        else:
            self.bert = pretrain_bert_uncased_model()
            self.text_dim = 768
        
        self.img_dim = 1024
        self.hubert_dim = 1024
        self.fea_dim = 128
        
        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, self.trans_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, self.trans_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_audio = nn.Sequential(torch.nn.Linear(self.hubert_dim, self.trans_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        
        self.tfr_module = CMFR(model_dimension=self.trans_dim,heads=self.num_heads,n_tokens=self.n_tokens,dropout=self.dropout)
        # self.tca_module = TokenLevelCrossModalAlignment(temperature=0.07)
        self.sca_module = CrossModalAlignment(d_model=self.trans_dim, n_heads=self.num_heads)
        self.UnifiedTransformer = UnifiedTransformer(d_model=self.trans_dim, num_layers=1, num_heads=self.num_heads, d_ff=self.trans_dim*2, dropout=self.dropout)
        self.pool_learner = AttnPooling(feat_dim=self.trans_dim, compressed_dim=self.fea_dim)
        
        self.classfire = nn.Linear(self.fea_dim, 2)
        # self.classfire = nn.Linear(self.trans_dim, 2)


    def forward(self, **kwargs):
        # text
        title_inputid = kwargs['title_inputid']  # (batch,512)
        title_mask = kwargs['title_mask']  # (batch,512)
        fea_text = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']  # (batch,sequence,dim)
        fea_text = self.linear_text(fea_text)
        
        # visual
        frames = kwargs['frames']  
        fea_img = self.linear_img(frames)
        
        # audio
        audioframes = kwargs['audio_feas']  
        fea_audio = self.linear_audio(audioframes)
        
        # Text-guided Feature Refinement module
        fea_text, fea_img, fea_audio = self.tfr_module(fea_text, fea_img, fea_audio)
        
        # CrossModalAlignment
        # loss = self.tca_module(fea_text, fea_img, fea_audio)
        fea_text, fea_img, fea_audio = self.sca_module(fea_text, fea_img, fea_audio)
        
        # UnifiedTransformer
        mixed_fea = torch.cat((fea_text, fea_img, fea_audio), dim=1)
        mixed_fea = self.UnifiedTransformer(mixed_fea)
        fea = self.pool_learner(mixed_fea)
        # fea = torch.mean(mixed_fea,dim=1).squeeze()
        output = self.classfire(fea)
        return output



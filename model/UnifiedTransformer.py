import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True, )
        std = z.std(dim=-1, keepdim=True, )
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))

        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return self.layer_norm(residual + output)


class FeedForward(nn.Module):
    def __init__(self, model_dimension, d_ff, dropout=0.1):
        super().__init__()
        self.ff1 = nn.Linear(model_dimension, d_ff)
        self.ff2 = nn.Linear(d_ff, model_dimension)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dimension)

    def forward(self, representations_batch):
        return self.norm(representations_batch + self.ff2(self.dropout(F.relu(self.ff1(representations_batch)))))

class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)
    def forward(self, srb1, srb2, mha):
        return srb1 + self.dropout(mha(self.norm(srb1), self.norm(srb2)))


class MultiHeadedAttentionAMA(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights, num_modalities=3):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads
        self.num_modalities = num_modalities

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)
        # self.softmax = nn.Softmax(dim=-1) 

        self.log_attention_weights = log_attention_weights
        self.attention_weights = None
        
        # 初始化权重
        self.modality_weights_raw = nn.Parameter(torch.zeros(num_modalities))

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        modality_len = seq_len // self.num_modalities
        
        # 标准 QKV 投影和分割 heads
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]
        
        # 计算原始注意力分数: [batch, heads, q_seq_len, k_seq_len (3*l)]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        
        # 分模态 Softmax
        split_scores = torch.split(scores, modality_len, dim=-1)
        norm_scores_list = [torch.softmax(mod_score, dim=-1) for mod_score in split_scores]
        attention_weights_AMA = torch.cat(norm_scores_list, dim=-1)
        
        # 获取数值稳定的权重
        actual_weights = torch.exp(self.modality_weights_raw)
        # print(actual_weights)

        # 应用可学习的模态权重
        weights_per_token = torch.repeat_interleave(actual_weights, modality_len)
        attn_scale_weights = weights_per_token.view(1, 1, 1, seq_len) 
        
        attention_weights = attention_weights_AMA * attn_scale_weights
        
        # 重新归一化
        attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        attention_weights = self.attention_dropout(attention_weights)
        intermediate_token_representations = torch.matmul(attention_weights, value)
        
        if self.log_attention_weights:
            self.attention_weights = attention_weights.detach()

        # 合并 heads
        reshaped = intermediate_token_representations.transpose(1, 2).contiguous().view(
            batch_size, -1, self.number_of_heads * self.head_dimension
        )
        # 输出投影
        token_representations = self.out_projection_net(reshaped)
        
        return token_representations

    def get_modality_weights(self):
        return torch.exp(self.modality_weights_raw).detach().cpu().numpy()
    
def get_clones(module, num_of_deep_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


class UnifiedTransformer(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # 构建编码器层
        self.attentions = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        # self.ffn_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # self.attentions.append(MultiHeadedAttention(d_model, num_heads, dropout, log_attention_weights=False))
            self.attentions.append(MultiHeadedAttentionAMA(d_model, num_heads, dropout, log_attention_weights=False))
            self.attn_norms.append(LayerNormalization(d_model))
            self.ffns.append(FeedForward(d_model, d_ff))
            # self.ffn_norms.append(LayerNormalization(d_model))
        
        self.dropout = nn.Dropout(dropout)
    
    
        # self.init_params()
        # self.norm = nn.LayerNorm(model_dimension)

    # def init_params(self, default_initialization=False):
    #     if not default_initialization:
    #         # model.named_parameters
    #         for name, p in self.named_parameters():
    #             if p.dim() > 1:
    #                 nn.init.xavier_uniform_(p)
    
    
    def forward(self, x):
        for i in range(self.num_layers):
            # 多头注意力/残差/层归一化
            attn_out = self.attentions[i](x, x, x)
            x = self.attn_norms[i](x + self.dropout(attn_out))
            
            # ffn/残差/层归一化
            ffn_out = self.ffns[i](x)
            # x = self.ffn_norms[i](x + self.dropout(moe_out))
        
        return x
    
# val
def test():
    batch_size = 4
    s_len = 12
    model_dim = 128
    n_experts = 2
    heads = 4
    top_k = 1
    dropout = 0.1
    transformer = UnifiedTransformer(d_model=model_dim, num_layers=1, num_heads=heads, d_ff=model_dim, dropout=0.1).cpu()
    
    features = torch.randn(batch_size, s_len, model_dim).cpu()
    
    print(f"输入形状:")
    print(f"  features: {features.shape}")
    
    # model.eval()
    transformer.eval()
    with torch.no_grad():
        # output = model(features)
        output = transformer(features)
    
    print(f"\n输出形状:")
    print(f"  output: {output.shape}")

# test()
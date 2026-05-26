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


class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability) 
        self.softmax = nn.Softmax(dim=-1) 

        self.log_attention_weights = log_attention_weights  
        self.attention_weights = None  

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        attention_weights = self.softmax(scores)
        attention_weights = self.attention_dropout(attention_weights)
        intermediate_token_representations = torch.matmul(attention_weights, value)
        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]

        intermediate_token_representations, attention_weights = self.attention(query, key, value)

        if self.log_attention_weights:
            self.attention_weights = attention_weights
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1,
                                                                              self.number_of_heads * self.head_dimension)
        # forward
        token_representations = self.out_projection_net(reshaped)
        return token_representations

def get_clones(module, num_of_deep_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


# Feature Refinement module
class CMFR(nn.Module):
    def __init__(self, n_tokens, model_dimension, heads, dropout):
        super().__init__()
        # trainable_token
        self.denoising_tokens_text = nn.Parameter(torch.randn(1, n_tokens, model_dimension))
        self.denoising_tokens_vision = nn.Parameter(torch.randn(1, n_tokens, model_dimension))
        self.denoising_tokens_audio = nn.Parameter(torch.randn(1, n_tokens, model_dimension))
        
        # modality Token 
        self.spec_tokens_text = nn.Parameter(torch.randn(1, n_tokens // 2, model_dimension))
        self.spec_tokens_vision = nn.Parameter(torch.randn(1, n_tokens // 2, model_dimension))
        self.spec_tokens_audio = nn.Parameter(torch.randn(1, n_tokens // 2, model_dimension))

        # share Token
        self.shared_tokens = nn.Parameter(torch.randn(1, n_tokens // 2, model_dimension))
        
        self.text_compress_sublayer = SublayerLogic(model_dimension, dropout)
        self.text_compress_mha = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=heads,
                                         dropout_probability=dropout, log_attention_weights=False)
        
        self.video_compress_sublayer = SublayerLogic(model_dimension, dropout)
        self.video_compress_mha = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=heads,
                                         dropout_probability=dropout, log_attention_weights=False)
        
        self.audio_compress_sublayer = SublayerLogic(model_dimension, dropout)
        self.audio_compress_mha = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=heads,
                                         dropout_probability=dropout, log_attention_weights=False)

        # self.text_ffn_layer = PoswiseFeedForwardNet(model_dimension, model_dimension * 2)
        # self.video_ffn_layer = PoswiseFeedForwardNet(model_dimension, model_dimension * 2)
        # self.audio_ffn_layer = PoswiseFeedForwardNet(model_dimension, model_dimension * 2)
        self.text_ffn_layer = FeedForward(model_dimension, model_dimension * 2)
        self.video_ffn_layer = FeedForward(model_dimension, model_dimension * 2)
        self.audio_ffn_layer = FeedForward(model_dimension, model_dimension * 2)
        
        self.init_params()
        # self.norm = nn.LayerNorm(model_dimension)

    def init_params(self, default_initialization=False):
        if not default_initialization:
            # model.named_parameters
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, text_input, video_input, audio_input):
        batch_size = text_input.size(0)
        
        # concat
        D_t = torch.cat([self.spec_tokens_text.expand(batch_size, -1, -1), 
                         self.shared_tokens.expand(batch_size, -1, -1)], dim=1)

        D_v = torch.cat([self.spec_tokens_vision.expand(batch_size, -1, -1), 
                         self.shared_tokens.expand(batch_size, -1, -1)], dim=1)

        D_a = torch.cat([self.spec_tokens_audio.expand(batch_size, -1, -1), 
                         self.shared_tokens.expand(batch_size, -1, -1)], dim=1)
    
        # 扩展 trainable token
        # D_t = self.denoising_tokens_text.expand(batch_size, -1, -1)
        # D_v = self.denoising_tokens_vision.expand(batch_size, -1, -1)
        # D_a = self.denoising_tokens_audio.expand(batch_size, -1, -1)

        # textual modality
        compress_attn_t = lambda x, y: self.text_compress_mha(query=x, key=y, value=y)
        D_t = self.text_compress_sublayer(D_t, text_input, compress_attn_t)
        
        # FFN
        D_t = self.text_ffn_layer(D_t)

        # visual modality
        compress_attn_v = lambda x, y: self.video_compress_mha(query=x, key=y, value=y)
        D_v = self.video_compress_sublayer(D_v, video_input, compress_attn_v)

        # FFN
        D_v = self.video_ffn_layer(D_v)

        # audio
        compress_attn_a = lambda x, y: self.audio_compress_mha(query=x, key=y, value=y)
        D_a = self.audio_compress_sublayer(D_a, audio_input, compress_attn_a)
        
        # FFN
        D_a = self.audio_ffn_layer(D_a)

        # 输出
        # R_t = self.norm(D_t)
        # R_v = self.norm(D_v)
        # R_a = self.norm(D_a)

        # return R_t, R_v, R_a
        return D_t, D_v, D_a

    
# val
def test_tfr():
    batch_size = 4
    text_len = 50
    vision_len = 32
    audio_len = 40
    model_dim = 512
    n_tokens = 8
    heads = 8
    dropout = 0.1
    
    model = CMFR(
        model_dimension=model_dim,
        heads=heads,
        n_tokens=n_tokens,
        dropout=dropout
    )
    
    text_features = torch.randn(batch_size, text_len, model_dim)
    vision_features = torch.randn(batch_size, vision_len, model_dim)
    audio_features = torch.randn(batch_size, audio_len, model_dim)
    
    print(f"输入形状:")
    print(f"  text_features: {text_features.shape}")
    print(f"  vision_features: {vision_features.shape}")
    print(f"  audio_features: {audio_features.shape}")
    
    model.eval()
    with torch.no_grad():
        R_t, R_v, R_a = model(text_features, vision_features, audio_features)
    
    print(f"\n输出形状:")
    print(f"  R_t: {R_t.shape}")
    print(f"  R_v: {R_v.shape}")
    print(f"  R_a: {R_a.shape}")

# test_tfr()


import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAlignment(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(CrossModalAlignment, self).__init__()
        self.d_model = d_model
        
        # projection
        self.hub_t_proj = nn.Linear(d_model, d_model)
        self.hub_v_proj = nn.Linear(d_model, d_model)
        self.hub_a_proj = nn.Linear(d_model, d_model)
        
        # Gathering
        self.gather_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gather_norm = nn.LayerNorm(d_model)
        
        # Broadcasting
        self.broadcast_attn_t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.broadcast_attn_v = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.broadcast_attn_a = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)

    def forward(self, text_features, visual_features, audio_features):
        b, l, d = text_features.shape
        
        #  Hub Tokens generate
        # h_t = self.hub_t_proj(text_features.mean(dim=1, keepdim=True))
        # h_v = self.hub_v_proj(visual_features.mean(dim=1, keepdim=True))
        # h_a = self.hub_a_proj(audio_features.mean(dim=1, keepdim=True))
        
        h_t = self.hub_t_proj(text_features.mean(1, keepdim=True) + text_features.max(1, keepdim=True)[0])
        h_v = self.hub_v_proj(visual_features.mean(1, keepdim=True) + visual_features.max(1, keepdim=True)[0])
        h_a = self.hub_a_proj(audio_features.mean(1, keepdim=True) + audio_features.max(1, keepdim=True)[0])
        
        hub = torch.cat([h_t, h_v, h_a], dim=1)
        
        # Gathering
        concat_memory = torch.cat([text_features, visual_features, audio_features], dim=1)
        
        # 3个 Hub Token 去看拼接后特征，更新语义
        updated_hub, _ = self.gather_attn(query=hub, key=concat_memory, value=concat_memory)
        updated_hub = self.gather_norm(hub + updated_hub) 
        
        # testing
        std_init = hub.std(dim=-1).mean().item()
        std_updated = updated_hub.std(dim=-1).mean().item()
        sample_sim = F.cosine_similarity(updated_hub[0:1], updated_hub[1:2], dim=-1).mean().item() if b > 1 else 0
        # print(f"--- Alignment Debug ---")
        # print(f"Hub Init Std: {std_init:.4f} | Updated Hub Std: {std_updated:.4f}")
        # print(f"Inter-sample Similarity: {sample_sim:.4f}")
        # print(f"-----------------------")
        
        # Broadcasting
        # t
        aligned_t_attn, _ = self.broadcast_attn_t(query=text_features, key=updated_hub, value=updated_hub)
        aligned_t = self.norm_t(text_features + aligned_t_attn)
        
        # v
        aligned_v_attn, _ = self.broadcast_attn_v(query=visual_features, key=updated_hub, value=updated_hub)
        aligned_v = self.norm_v(visual_features + aligned_v_attn)
        
        # a
        aligned_a_attn, _ = self.broadcast_attn_a(query=audio_features, key=updated_hub, value=updated_hub)
        aligned_a = self.norm_a(audio_features + aligned_a_attn)
        
        return aligned_t, aligned_v, aligned_a

    
    
# if __name__ == "__main__":
#     batch_size, seq_len, dim = 32, 50, 256
#     model = CrossModalAlignment(d_model=dim, n_heads=8)
    
#     T = torch.randn(batch_size, seq_len, dim)
#     V = torch.randn(batch_size, seq_len, dim)
#     A = torch.randn(batch_size, seq_len, dim)
    
#     out_T, out_V, out_A = model(T, V, A)
#     print(f"Output shapes: Text {out_T.shape}, Visual {out_V.shape}, Audio {out_A.shape}")
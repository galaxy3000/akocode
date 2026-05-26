import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLevelCrossModalAlignment(nn.Module):
    
    def __init__(self, temperature: float = 0.07, device: str = 'cuda'):
        super(TokenLevelCrossModalAlignment, self).__init__()
        self.temperature = temperature
        self.device = device
        
        self.v_align = nn.Linear(512, 512, bias=False)
        self.t_align = nn.Linear(512, 512, bias=False)
        self.a_align = nn.Linear(512, 512, bias=False)

#     def compute_similarity_score_batch(self, R_a, R_b) -> torch.Tensor:
#         # 计算所有 token 之间的点积矩阵 [b, n, n]
#         S = torch.bmm(R_a, R_b.transpose(1, 2))

#         # 直接对所有 token 对的相似度求平均，得到全局相似度 [b, 1]
#         # dim=(1, 2) 同时在两个模态的时间/空间维度上做平均
#         similarity_scores = S.mean(dim=(1, 2), keepdim=True)
#         similarity_scores = similarity_scores.squeeze(-1)
#         # print(similarity_scores.shape)
#         return similarity_scores

    def compute_similarity_score_batch(self, 
                                        R_a: torch.Tensor, 
                                        R_b: torch.Tensor) -> torch.Tensor:
        
        # 计算token-wise相似度矩阵
        # R_a: [b, n, d], R_b: [b, n, d] -> S: [b, n, n]
        # (4,16,512),(4,16,512)
        S = torch.bmm(R_a, R_b.transpose(1, 2))
        # print(f"S: min={S.min():.4f}, max={S.max():.4f}, mean={S.mean():.4f}")
        # print(S.shape)
        # (4,16,16)
        # 第一轮池化 - 获得token-sentence相似度向量
        # 对S的每个样本，沿着visual token维度做softmax
        attention_weights_1 = F.softmax(S / self.temperature, dim=2)  # [b, n, n]
        # print(f"attn1: min={attention_weights_1.min():.4f}, max={attention_weights_1.max():.4f}")
        # print(attention_weights_1.shape)
        # (4,16,16)
        # 加权求和得到S_tilde: [b, n, 1]
        S_tilde = torch.sum(attention_weights_1 * S, dim=2)
        # print(f"S_tilde: min={S_tilde.min():.4f}, max={S_tilde.max():.4f}, mean={S_tilde.mean():.4f}")
        # (4,1,16)
        # print(S_tilde.shape)
        # 第二轮池化 - 获得最终的细粒度相似度分数
        # S_tilde_squeezed = S_tilde.squeeze(-2)  # [b, n]
        # (4,1)
        # print(S_tilde_squeezed.shape)
        attention_weights_2 = F.softmax(S_tilde / self.temperature, dim=1)  # [b, n]
        
        # print(attention_weights_2.shape)
        similarity_scores = torch.sum(attention_weights_2 * S_tilde, dim=1, keepdim=True)  # [b, 1]
        # print(f"Final scores: {similarity_scores.squeeze()}")
        # print(similarity_scores.shape)
        # (4,1)
        return similarity_scores
    
    def forward(self, 
            text_features: torch.Tensor, 
            visual_features: torch.Tensor, 
            audio_features: torch.Tensor) -> torch.Tensor:
    
        visual_features = visual_features.to(self.device)
        audio_features = audio_features.to(self.device)
        text_features = text_features.to(self.device)
        
        batch_size = visual_features.shape[0]
        
        # L2 归一化 normalize
        # visual_features = F.normalize(visual_features, p=2, dim=-1)
        # audio_features = F.normalize(audio_features, p=2, dim=-1)
        # text_features = F.normalize(text_features, p=2, dim=-1)
        
        visual_features = F.normalize(self.v_align(visual_features), p=2, dim=-1)
        audio_features = F.normalize(self.t_align(audio_features), p=2, dim=-1)
        text_features = F.normalize(self.a_align(text_features), p=2, dim=-1)
        
        labels = torch.arange(batch_size, device=self.device)
        
        
        # 计算视觉-文本相似度矩阵
        sim_matrix_vt = torch.zeros(batch_size, batch_size, device=self.device)
        
        for i in range(batch_size):
            text_i_batch = text_features[i:i+1].expand(batch_size, -1, -1)
            scores = self.compute_similarity_score_batch(visual_features, text_i_batch).squeeze(-1)
            # print(scores)
            sim_matrix_vt[:, i] = scores
            #  # 当前文本样本
            # text_i = text_features[i:i+1]
            # # print(text_i.shape)
            # text_i = text_features[i:i+1].repeat(batch_size, 1, 1)   # [b, l, d]
            # # print(text_i.shape)
            # # print(visual_features.shape)
            # # 计算所有视觉样本与当前文本样本的相似度
            # sim_matrix_vt[:, i] = self.compute_similarity_score_batch(visual_features, text_i).squeeze(-1)
            # # print(sim_matrix_vt)
        # print(sim_matrix_vt)
        
        # 对列做 softmax
        loss_vt_t2i = F.cross_entropy(sim_matrix_vt.t(), labels)
        
        # 对行做 softmax
        # loss_vt_i2t = F.cross_entropy(sim_matrix_vt, labels)
        # loss_vt = (loss_vt_i2t + loss_vt_t2i) / 2
        
        loss_vt = loss_vt_t2i
        
        pos_sim = torch.diag(sim_matrix_vt).mean()
        neg_sim = (sim_matrix_vt.sum() - torch.diag(sim_matrix_vt).sum()) / (batch_size * batch_size - batch_size)
        # print(f"Positive similarity: {pos_sim:.4f}, Negative similarity: {neg_sim:.4f}")
        # print(f"Margin: {pos_sim - neg_sim:.4f}")
        # # 计算视觉-文本损失
        # exp_sim_vt = torch.exp(sim_matrix_vt)
        # # print(exp_sim_vt)
        # pos_sim_vt = exp_sim_vt.diag()
        # # print(pos_sim_vt)
        # sum_exp_sim_vt = exp_sim_vt.sum(dim=1)
        # # print(sum_exp_sim_vt)
        # loss_vt = -torch.log(pos_sim_vt / sum_exp_sim_vt).mean()
        
        # 计算音频-文本相似度矩阵
#         sim_matrix_at = torch.zeros(batch_size, batch_size, device=self.device)
        
#         for i in range(batch_size):
#             text_i = text_features[i:i+1].repeat(batch_size, 1, 1) 
#             sim_matrix_at[:, i] = self.compute_similarity_score_batch(audio_features, text_i).squeeze(-1)
#             # print(sim_matrix_at)
        
#         # 计算音频-文本损失
#         exp_sim_at = torch.exp(sim_matrix_at)
#         # print(exp_sim_at)
#         pos_sim_at = exp_sim_at.diag()
#         # print(pos_sim_at)
#         sum_exp_sim_at = exp_sim_at.sum(dim=1)
#         # print(sum_exp_sim_at)
#         loss_at = -torch.log(pos_sim_at / sum_exp_sim_at).mean()
        sim_matrix_at = torch.zeros(batch_size, batch_size, device=self.device)
        
        
        for i in range(batch_size):
            text_i_batch = text_features[i:i+1].expand(batch_size, -1, -1)
            scores = self.compute_similarity_score_batch(audio_features, text_i_batch).squeeze(-1)
            sim_matrix_at[:, i] = scores
        # print(sim_matrix_at)
        
        loss_at_t2i = F.cross_entropy(sim_matrix_at.t(), labels)
        
        # 音频检索文本 / 文本检索音频
        # loss_at_i2t = F.cross_entropy(sim_matrix_at, labels)
        # loss_at = (loss_at_i2t + loss_at_t2i) / 2
        
        loss_at = loss_at_t2i
        print(loss_vt)
        print(loss_at)
        total_loss = loss_vt + loss_at
        # print(total_loss)
        return total_loss


# 使用示例
# if __name__ == "__main__":
#     # 设置参数
#     batch_size = 4
#     seq_len = 16
#     feature_dim = 512
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # 创建随机输入
#     visual_features = torch.randn(batch_size, seq_len, feature_dim).to(device)
#     audio_features = torch.randn(batch_size, seq_len, feature_dim).to(device)
#     text_features = torch.randn(batch_size, seq_len, feature_dim).to(device)
    
#     # visual_features = F.normalize(visual_features, dim=-1)  # 注意是dim=-1
#     # audio_features = F.normalize(audio_features, dim=-1)
#     # text_features = F.normalize(text_features, dim=-1)
    
#     # 初始化TCA模块
#     tca = TokenLevelCrossModalAlignment(temperature=0.07, device=device)
    
#     # 计算损失
#     loss = tca(visual_features, audio_features, text_features)
    
#     print(f"Total Alignment Loss: {loss.item():.4f}")


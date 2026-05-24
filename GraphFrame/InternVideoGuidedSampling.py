import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig

class InternVideoGuidedSampling:
    """
    使用InternVideo2.5的全局特征 [T, 64, D] 指导FrameThinker选帧
    """
    def __init__(self, model, video_features, importance_scores, model_path="/home/teacher_tang/Video-Holmes/InternVL_2_5_HiCo_R64_series/InternVL_2_5_HiCo_R64"):
        self.model = model
        self.video_features = video_features  # [T, 64, 1408]sss
        self.importance_scores = importance_scores  # [T]
        self.selected_frames = set()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
         
    def suggest_initial_frames(self, fps_1_frame_indices, num_frames=8):
        # fps_1_frame_indices 是视频中真正的帧索引，假如取600帧，那么importance_scores就是0到600，但600在视频中真正的帧索引中可能对应1200，与视频原本的fps有关
        # print(fps_1_frame_indices)
        T = len(self.importance_scores)
        # Top-K重要帧
        topk_indices = torch.topk(self.importance_scores, k=num_frames//2).indices
        topk_frames_index = []
        for i in topk_indices:
            topk_frames_index.append(fps_1_frame_indices[i])

        # 均匀采样
        uniform_indices = torch.linspace(0, T-1, num_frames//2).long()
        uniform_frames_index = []
        for i in uniform_indices:
            uniform_frames_index.append(fps_1_frame_indices[i])
        
        initial_frames = set(list(topk_frames_index) + list(uniform_frames_index))
        mapping_frames = set(list(topk_indices) + list(uniform_indices)) # 这个是映射后的，即0到600
        self.selected_frames.update(list(mapping_frames))
        return list(initial_frames)
    
    def suggest_next_frames(self, query, trajectory, fps, fps_1_frame_indices, num_frames=4):
        # 计算查询相关性：Query[1408] 与 Video[T, 64, 1408] 匹配
        query_relevance = self.compute_query_relevance(query)
        
        # 计算图谱覆盖度
        graph_coverage = self.compute_graph_coverage(trajectory, fps, fps_1_frame_indices)
        
        # 计算未观察奖励
        unexplored_bonus = torch.ones(len(self.importance_scores)).to(self.video_features.device)
        for idx in self.selected_frames:
            unexplored_bonus[idx] = 0.1
        
        # 综合打分
        scores = (
            0.4 * query_relevance + 
            0.3 * (1 - graph_coverage) + 
            0.2 * self.importance_scores + 
            0.1 * unexplored_bonus
        )
        
        candidate_indices = torch.topk(scores, k=min(num_frames*3, len(scores))).indices
        selected = self.diversity_sampling(candidate_indices, num_frames)
        selected_frames_index = []
        for i in selected:
            selected_frames_index.append(fps_1_frame_indices[i])
        self.selected_frames.update(selected)
        return selected_frames_index
    
    def compute_query_relevance(self, query):
        """
        计算 Query [D] 与 Frame Tokens [64, D] 的最大语义匹配度
        """
        # query_embedding: [D] -> [1, 1, D]
        # video_features: [T, 64, D]
        
        inputs = self.tokenizer(query, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids'] # [1, L]
        
        question_embeds = self.model.language_model.get_input_embeddings()(input_ids) # [1, L, 4096]
        
        # 池化处理 (Pooling), 视频特征是 [T, 64, 4096]，Query 需要是一个 [4096] 的向量, 推荐使用 Mean Pooling（均值池化），因为它能捕捉问题的整体语义
        query_embedding = question_embeds.mean(dim=1).squeeze(0) # [4096]
        
        # 归一化 (可选，但推荐), InternVideoGuidedSampling 中计算余弦相似度时，归一化能让分数更稳定
        query_embedding = F.normalize(query_embedding, p=2, dim=0)

        # 对每一帧的 64 个 Token 进行计算，取最大相似度（表示该帧内只要有任何局部区域匹配就算相关）
        # 归一化特征
        v_feat = F.normalize(self.video_features, p=2, dim=2) # [T, 64, D]
        q_feat = F.normalize(query_embedding, p=2, dim=0).view(1, 1, -1) # [1, 1, D]
        
        # 点积相似度: [T, 64]
        sim_matrix = torch.matmul(v_feat, q_feat.transpose(1, 2)).squeeze(-1)
        
        # 对 64 个 Token 取 Max-Pooling，捕捉最相关的局部物体
        query_relevance = sim_matrix.max(dim=1).values
        return query_relevance
    
    def compute_graph_coverage(self, trajectory, fps, fps_1_frame_indices):
        T = len(self.video_features)
        fps_1_frame_second = []
        for i in fps_1_frame_indices:
            fps_1_frame_second.append(float(i) / float(fps))
        fps_1_frame_second_copy = fps_1_frame_second

        graph_coverage = torch.zeros(T).to(self.video_features.device)
        for tra in trajectory:
            start, end = tra['subgraph']['timestamp_range']  
            for se in range(len(fps_1_frame_second_copy)-1):
                if start <= fps_1_frame_second_copy[se] <= end:
                    graph_coverage[se] = 1.0
                else:
                    graph_coverage[se] = 0.0
        return graph_coverage
    
    def diversity_sampling(self, candidate_indices, num_frames):
        selected = []
        candidate_indices = candidate_indices.tolist()
        while len(selected) < num_frames and candidate_indices:
            best_idx = candidate_indices.pop(0)
            if not selected or min(abs(best_idx - s) for s in selected) > 30:
                selected.append(best_idx)
        return selected
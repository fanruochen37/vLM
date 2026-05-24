import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'
import json
from tqdm import tqdm
import time
import random
import csv
from collections import defaultdict
import math
import numpy as np
import re
from PIL import Image
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig, PretrainedConfig
import os
import sys
from InternVL_2_5_HiCo_R64_series.evaluate import *


# 彻底封印 transformers 的日志和 repr 逻辑
import transformers.configuration_utils

# 禁用 transformers 的日志，防止它尝试打印 config
transformers.utils.logging.set_verbosity_error()

# 暴力重写 PretrainedConfig 的 __repr__，彻底关掉引发报错的入口，当 logger.info(f"{config}") 调用时，直接返回空字符串，不再走 to_diff_dict()
def fast_repr(self):
    return f"{self.__class__.__name__} (Debug: Repr disabled to prevent InternVL Bug)"

transformers.configuration_utils.PretrainedConfig.__repr__ = fast_repr


def apply_internvl_patch():
    """
    由于 InternVLChatConfig 在无参构造时不会初始化 llm_config，
    导致其 to_dict 方法报错。此补丁在内存中动态修正其 to_dict 方法。
    """
    # 查找内存中已经加载的 InternVL 配置模块
    # 注意：路径可能因 transformers 版本或缓存位置略有不同，这里覆盖常见命名
    target_modules = [
        'transformers_modules.InternVL_2_5_HiCo_R64.configuration_internvl_chat',
        'configuration_internvl_chat'
    ]
    
    for module_name in target_modules:
        if module_name in sys.modules:
            config_class = getattr(sys.modules[module_name], 'InternVLChatConfig', None)
            if config_class:
                # 获取原始的 to_dict
                original_to_dict = config_class.to_dict
                
                # 定义新的安全 to_dict
                def safe_to_dict(self):
                    # 如果当前实例缺失 llm_config（通常是 transformers 内部自检时产生的空实例）
                    # 手动补上一个基础配置，防止 self.llm_config.to_dict() 崩溃
                    if not hasattr(self, 'llm_config') or self.llm_config is None:
                        # 补一个空的 PretrainedConfig，它有自己的 to_dict 方法
                        self.llm_config = PretrainedConfig()
                    
                    # 额外检查 vision_config 确保万无一失
                    if not hasattr(self, 'vision_config') or self.vision_config is None:
                        self.vision_config = PretrainedConfig()
                        
                    return original_to_dict(self)
                
                # 替换方法
                config_class.to_dict = safe_to_dict
                break


class InternVideoEncoder:
    def __init__(self, config, encoder):

        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        self.config = config
        apply_internvl_patch()
        if not hasattr(self.config, 'llm_config') or self.config.llm_config is None:
            if hasattr(self.config, 'sub_configs'):
                self.config.llm_config = self.config.sub_configs.get('llm_config', PretrainedConfig())
            else:
                self.config.llm_config = PretrainedConfig()
        self.encoder = encoder
        self.encoder.eval()
        
    def load_video(self, video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values) # 按帧顺序加入
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list, frame_indices, fps

    def encode_full_video(self, video_path, fps=1, local_num_frames=4):
        """
        输入: 视频路径
        输出: [T, 64, 1408] 的特征矩阵
        """

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        duration = total_frames / video_fps
        target_frame_count = int(duration * fps)
        target_frame_count = math.floor(target_frame_count / local_num_frames) * local_num_frames
        pixel_values, num_patches_list, frame_indices, fps = self.load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=target_frame_count)

        with torch.no_grad():
            video_features = self.encoder.extract_feature(pixel_values.to(torch.bfloat16).to(next(self.encoder.parameters()).device)) 
            print(video_features.shape)
            
        # 时序建模：保持 64 tokens 独立，仅在时间轴平滑
        video_features = self.temporal_modeling(video_features)
        return video_features, frame_indices
    
    def temporal_modeling(self, video_features):
        """
        轻量级时序建模：在 [T, 64, D] 上进行时间轴平滑
        """
        T, N, D = video_features.shape # [600, 64, 1408]
        # 转置为 [64, D, T] 以便对时间轴做卷积
        x = video_features.permute(1, 2, 0) 
        
        kernel_size = 5
        # 对每一个 Token 位置独立进行时间轴平滑
        x_smoothed = F.avg_pool1d(
            x, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size//2
        )
        
        # 还原回 [T, 64, D]
        return x_smoothed.permute(2, 0, 1)
    
    def get_temporal_importance(self, video_features):
        """
        基于 Token 变化的运动检测
        """
        # 帧间差异：计算相邻帧对应 Token 的欧氏距离变化
        # video_features: [T, 64, D]
        
        # 计算每一帧 64 个 token 的均值，用于代表该帧整体能量
        frame_mean = video_features.mean(dim=1) # [T, D]
        
        feature_diff = torch.diff(frame_mean, dim=0)
        importance_motion = torch.norm(feature_diff, dim=1)
        # 补充第一帧的分数
        importance_motion = torch.cat([importance_motion[0:1], importance_motion])
        
        # 局部 Token 差异方差：检测画面内部是否有剧烈局部变动（如物体闪现）
        token_variance = video_features.var(dim=1).mean(dim=1) # [T]
        
        importance_scores = 0.7 * importance_motion + 0.3 * token_variance
        return importance_scores



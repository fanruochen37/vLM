import os
import json
from typing import List, Dict, Optional, Union
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from nano_graphrag import GraphRAG, QueryParam


class Vision2GraphParser:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
        """
        初始化视觉语言模型用于帧分析
        """
        print(f"加载视觉语言模型: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.vlm.eval()
        print("视觉语言模型加载完成")

    def describe_frame(self, frame_path: str, prompt: str) -> str:
        """
        使用Qwen2VL描述单帧图像
        
        Args:
            frame_path: 帧图像路径
            prompt: 分析提示词
            
        Returns:
            str: 生成的描述文本
        """
        try:
            # 加载图像
            if not os.path.exists(frame_path):
                return f"图像文件不存在: {frame_path}"
            
            image = Image.open(frame_path).convert('RGB')
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image
                        },
                        {
                            "type": "text", 
                            "text": prompt
                        }
                    ]
                }
            ]
            
            # 预处理
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            
            # 将输入移动到模型所在的设备
            inputs = {k: v.to(self.vlm.device) for k, v in inputs.items()}
            
            # 生成描述
            with torch.no_grad():
                generated_ids = self.vlm.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.7
                )
                
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # 提取模型回复部分
            if "assistant" in generated_text:
                generated_text = generated_text.split("assistant")[-1].strip()
            
            return generated_text
            
        except Exception as e:
            return f"帧分析错误: {str(e)}"


class VideoToGraphRAG:
    """
    将视频帧转换为知识图谱的完整解决方案
    """
    
    def __init__(self, 
                 working_dir: str = "./video_graphrag",
                 using_azure_openai: bool = False,
                 using_amazon_bedrock: bool = False,
                 vl_model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        """
        初始化视频到知识图谱转换器
        
        Args:
            working_dir: 工作目录，用于存储图谱和中间文件
            vl_model_name: 视觉语言模型名称
        """
        self.working_dir = Path(working_dir)
        
        # 创建目录
        self.metadata_dir = self.working_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 GraphRAG     ###################可能需要改#####################
        self.graphrag = GraphRAG(
            working_dir=str(self.working_dir / "graphrag"),
            using_azure_openai=using_azure_openai,
            using_amazon_bedrock=using_amazon_bedrock
        )
        
        # 初始化视觉语言模型
        self.vision_parser = Vision2GraphParser(vl_model_name)
        
        # 存储元数据
        self.metadata = {
            "video_info": {},
            "events": [],
            "integrated_data": [],
            "entity_timelines": {}
        }
    
    def format_timestamp(self, seconds: float, mode: str = "detailed") -> str:
        """
        将秒数格式化为可读时间字符串
        
        Args:
            seconds: 秒数
            mode: 格式模式 - "simple"/"detailed"/"human"
            
        Returns:
            格式化后的时间字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        if mode == "simple":
            return f"{minutes:02d}:{secs:02d}"
        elif mode == "detailed":
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
            else:
                return f"{minutes:02d}:{secs:02d}.{milliseconds:03d}"
        elif mode == "human":
            if hours > 0:
                return f"{hours}小时{minutes}分{secs}秒"
            elif minutes > 0:
                return f"{minutes}分{secs}秒"
            else:
                return f"{secs}秒"
        else:
            return f"{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    def process_frames(self, frames_data: List[Dict]) -> List[Dict]:
        """
        处理提供的帧数据，确保格式正确
        
        Args:
            frames_data: 帧数据列表，每个字典应包含:
                - frame_id: 帧编号
                - timestamp: 时间戳（秒）
                - file_path: 帧图片路径
                
        Returns:
            List[Dict]: 标准化后的帧数据
        """
        print("处理帧数据...")
        
        processed_frames = []
        
        for i, frame_data in enumerate(frames_data):
            # 确保必需的字段存在
            if 'frame_id' not in frame_data:
                frame_data['frame_id'] = i
            if 'timestamp' not in frame_data:
                frame_data['timestamp'] = i * 5.0  # 默认每5秒一帧
            
            # 生成格式化时间戳
            if 'timestamp_str' not in frame_data:
                frame_data['timestamp_str'] = self.format_timestamp(frame_data['timestamp'])
            
            if 'extraction_reason' not in frame_data:
                frame_data['extraction_reason'] = 'provided'
            
            # 验证文件路径
            if 'file_path' not in frame_data:
                print(f"警告: 帧 {i} 缺少 file_path")
                continue
                
            if not os.path.exists(frame_data['file_path']):
                print(f"警告: 帧文件不存在: {frame_data['file_path']}")
                continue
            
            processed_frames.append(frame_data)
        
        print(f"帧数据处理完成: 共 {len(processed_frames)} 帧")
        return processed_frames
    
    def extract_audio_and_subtitles(video_path: str) -> Dict:
        """
        提取音频转录和字幕
        """
        import whisper
        from pysrt import open as open_srt
        
        # 语音转文本
        model = whisper.load_model("base")
        audio_result = model.transcribe(video_path)
        
        # 时间对齐的转录结果
        segments = []
        for segment in audio_result["segments"]:
            segments.append({
                "start": segment["start"],
                "end": segment["end"], 
                "text": segment["text"],
                "type": "speech"
            })
        
        # 如果有外部字幕文件
        subtitle_data = []
        try:
            subs = open_srt(video_path.replace('.mp4', '.srt'))
            for sub in subs:
                subtitle_data.append({
                    "start": sub.start.seconds,
                    "end": sub.end.seconds,
                    "text": sub.text,
                    "type": "subtitle"
                })
        except:
            pass
        
        return {
            "audio_segments": segments,
            "subtitles": subtitle_data
        }
    
    def analyze_visual_content(self, frames_data: List[Dict]) -> List[Dict]:
        """
        使用Qwen2VL分析帧视觉内容
        
        Args:
            frames_data: 帧数据列表
            
        Returns:
            List[Dict]: 增强的帧分析数据
        """
        print("开始使用Qwen2VL分析视觉内容...")
        
        analyzed_frames = []
        
        for i, frame_data in enumerate(frames_data):
            frame_path = frame_data["file_path"]
            
            # 基础视觉描述提示词
            basic_prompt = """
请详细描述这张图像的内容，包括：
1. 场景类型和环境特征
2. 可见的人物、物体和动物
3. 主要活动和动作
4. 色彩、光照和氛围
5. 图像中的重要细节
请提供全面且结构化的描述。
            """.strip()
            
            # 使用Qwen2VL生成基础描述
            basic_description = self.vision_parser.describe_frame(frame_path, basic_prompt)

            # 详细分析（使用文本LLM增强）
            detailed_analysis = self.enhance_visual_analysis(frame_path
                basic_description, 
                frame_data["timestamp_str"]
            )
            
            # 实体和关系提取提示词
            entity_prompt = """
请从这张图像中提取以下结构化信息：

## 实体识别
- 人物：姓名/特征、动作、表情、位置
- 物体：名称、功能、状态、位置  
- 动物：种类、行为、状态
- 地点：环境类型、特征

## 关系分析
- 实体之间的空间关系
- 实体之间的交互关系
- 动作和活动的关联

## 场景理解
- 当前发生的主要事件
- 情感氛围和情绪状态
- 潜在的故事进展

请以结构化的JSON格式返回分析结果，包含entities字段和relationships字段。
            """.strip()
            
            # 使用Qwen2VL提取实体和关系
            entity_analysis = self.vision_parser.describe_frame(frame_path, entity_prompt)  ##################要改########################
            
            # 解析实体信息（简化处理，实际可以更复杂）
            entities = self._parse_entities_from_analysis(entity_analysis, frame_data['frame_id'])
            
            analyzed_frames.append({
                **frame_data,
                "basic_description": basic_description,
                "detailed_analysis": detailed_analysis,
                "entities": entities,
                "visual_metadata": {
                    "file_path": frame_path,
                    "file_size": os.path.getsize(frame_path) if os.path.exists(frame_path) else 0,
                    "analyzed_by": "Qwen2-VL"
                }
            })
            
            # 进度显示
            if (i + 1) % 5 == 0 or (i + 1) == len(frames_data):
                print(f"已分析 {i + 1}/{len(frames_data)} 帧...")
        
        print("视觉内容分析完成")
        return analyzed_frames
    
    def enhance_visual_analysis(self, frame_path:str, basic_desc: str, timestamp: str) -> str:
        """
        使用文本LLM增强视觉分析
        """
        enhancement_prompt = f"""
        作为专业视频分析师，请详细描述以下画面：
        
        时间点: {timestamp}
        基础描述: {basic_desc}
        
        请按以下结构提供详细分析：
        
        ## 场景概览
        - 主要场景类型（室内/室外/特写等）
        - 整体氛围和色调
        
        ## 实体识别
        - 人物：数量、特征、动作、表情
        - 物体：重要物体、位置关系
        - 环境：地点特征、背景元素
        
        ## 活动分析
        - 正在发生的主要活动
        - 实体间的交互关系
        - 动态元素的状态
        
        ## 语义信息
        - 潜在的情节进展
        - 情感氛围
        - 可能的重要性和意义
        
        详细分析：
        """
        
        # 调用文本LLM（如DeepSeek）
        try:
            response = self.vision_parser.describe_frame(frame_path, enhancement_prompt)  
            return response
        except:
            return basic_desc  # 降级处理
    
    def _parse_entities_from_analysis(self, analysis_text: str, frame_id: int) -> List[Dict]:
        """
        从Qwen2VL的分析文本中解析实体信息
        
        Args:
            analysis_text: 分析文本
            frame_id: 帧编号
            
        Returns:
            List[Dict]: 解析出的实体列表
        """
        entities = []
        
        # 简化的实体解析逻辑
        # 在实际应用中，可以使用更复杂的NLP技术或要求模型返回结构化数据
        
        # 检查是否包含人物相关关键词
        person_keywords = ['人', '人物', '男人', '女人', '孩子', '人物', '人脸', '人物特征']
        if any(keyword in analysis_text for keyword in person_keywords):
            entities.append({
                "name": f"人物_{frame_id}",
                "type": "PERSON",
                "action": self._extract_action(analysis_text),
                "description": self._extract_entity_description(analysis_text, "人物"),
                "confidence": 0.8
            })
        
        # 检查物体
        object_keywords = ['物体', '物品', '工具', '设备', '家具', '车辆']
        if any(keyword in analysis_text for keyword in object_keywords):
            entities.append({
                "name": f"物体_{frame_id}",
                "type": "OBJECT",
                "action": "存在",
                "description": self._extract_entity_description(analysis_text, "物体"),
                "confidence": 0.7
            })
        
        # 检查动物
        animal_keywords = ['动物', '宠物', '狗', '猫', '鸟', '鱼']
        if any(keyword in analysis_text for keyword in animal_keywords):
            entities.append({
                "name": f"动物_{frame_id}",
                "type": "ANIMAL",
                "action": self._extract_action(analysis_text),
                "description": self._extract_entity_description(analysis_text, "动物"),
                "confidence": 0.7
            })
        
        # 如果未检测到特定实体，添加通用实体
        if not entities:
            entities.append({
                "name": f"场景_{frame_id}",
                "type": "SCENE",
                "action": "展示",
                "description": analysis_text[:100] + "..." if len(analysis_text) > 100 else analysis_text,
                "confidence": 0.6
            })
        
        return entities
    
    def _extract_action(self, text: str) -> str:
        """从文本中提取动作描述"""
        action_keywords = {
            '站立': ['站立', '站着', '直立'],
            '行走': ['行走', '走路', '移动', '行进'],
            '坐': ['坐', '坐着', '就坐'],
            '说话': ['说话', '交谈', '讨论'],
            '观看': ['看', '观看', '注视', '观察'],
            '工作': ['工作', '操作', '使用', '处理']
        }
        
        for action, keywords in action_keywords.items():
            if any(keyword in text for keyword in keywords):
                return action
        
        return "存在"
    
    def _extract_entity_description(self, text: str, entity_type: str) -> str:
        """提取实体相关描述"""
        # 简化的描述提取，实际可以使用更复杂的NLP技术
        sentences = text.split('。')
        relevant_sentences = [s for s in sentences if entity_type in s]
        
        if relevant_sentences:
            return relevant_sentences[0][:200]  # 限制长度
        else:
            return text[:150] + "..." if len(text) > 150 else text
    
    def _find_matching_audio(self, audio_segments: List[Dict], timestamp: float) -> List[Dict]:
        """查找匹配时间点的音频片段"""
        matching = []
        for segment in audio_segments:
            if segment["start"] <= timestamp <= segment["end"]:
                matching.append(segment)
        return matching
    
    def _find_matching_subtitles(self, subtitles: List[Dict], timestamp: float) -> List[Dict]:
        """查找匹配时间点的字幕"""
        matching = []
        for sub in subtitles:
            if sub["start"] <= timestamp <= sub["end"]:
                matching.append(sub)
        return matching
    
    def integrate_multimodal_data(self, visual_data: List[Dict], audio_data: Dict) -> List[Dict]:
        """
        融合视觉和音频信息
        
        Args:
            visual_data: 视觉分析数据
            audio_data: 音频数据
            
        Returns:
            List[Dict]: 融合后的数据
        """
        print("开始多模态数据融合...")
        
        integrated_data = []
        
        for visual_frame in visual_data:
            frame_timestamp = visual_frame["timestamp"]
            
            # 找到对应时间段的音频/字幕
            matching_audio = self._find_matching_audio(
                audio_data["audio_segments"], frame_timestamp
            )
            matching_subs = self._find_matching_subtitles(
                audio_data["subtitles"], frame_timestamp
            )
            
            # 构建综合描述 ####################细看一下决定是否要改########################
            combined_parts = [f"【时间点: {visual_frame['timestamp_str']}】"]
            combined_parts.append("## 视觉内容分析")
            combined_parts.append(visual_frame["detailed_analysis"])
            
            if matching_audio:
                combined_parts.append("## 对话/旁白")
                for audio_seg in matching_audio:
                    combined_parts.append(f"- {audio_seg['text']}")
            
            if matching_subs:
                combined_parts.append("## 字幕信息")
                for sub in matching_subs:
                    combined_parts.append(f"- {sub['text']}")
            
            combined_description = "\n".join(combined_parts)
            
            integrated_frame = {
                **visual_frame,
                "audio_content": matching_audio,
                "subtitle_content": matching_subs,
                "combined_description": combined_description,
                "has_audio": len(matching_audio) > 0,
                "has_subtitle": len(matching_subs) > 0
            }
            
            integrated_data.append(integrated_frame)
        
        print("多模态数据融合完成")
        return integrated_data
    
    def _analyze_temporal_continuity(self, prev: Dict, current: Dict) -> str:
        """
        分析两个连续场景的时序关系
        
        Args:
            prev: 前一帧数据
            current: 当前帧数据
            
        Returns:
            str: 连续性分析文本
        """
        # 使用Qwen2VL进行连续性分析（基于文本描述）
        continuity_prompt = f"""
请分析这两个连续时间点的场景关系：

【前一场景 {prev['timestamp_str']}】
{prev['basic_description']}

【当前场景 {current['timestamp_str']}】
{current['basic_description']}

请分析：
1. 场景间的变化和连续性
2. 实体如何移动、出现或消失  
3. 活动的进展和演变
4. 时间线上的逻辑关系

请提供详细的连续性分析：
        """.strip()
        
        try:   ################改一个纯文本LLM########################
            response = llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": continuity_prompt}]
            )
            return response.choices[0].message.content
        except:
            return "场景连续性分析暂不可用"
        
    
    def build_temporal_narrative(self, integrated_data: List[Dict]) -> str:
        """
        构建带有时序关系的完整叙事文本
        
        Args:
            integrated_data: 融合后的数据
            
        Returns:
            str: 完整的时序叙事文本
        """
        print("构建时序叙事...")
        
        narrative_parts = ["视频内容完整分析报告 - 基于Qwen2-VL分析", "=" * 50]
        
        # 按时间排序
        sorted_data = sorted(integrated_data, key=lambda x: x["timestamp"])
        
        for i, current_frame in enumerate(sorted_data):
            narrative_parts.append(f"\n📊 场景 {i+1}: {current_frame['timestamp_str']}")
            narrative_parts.append(current_frame["combined_description"])
            
            # 添加与前一场景的连续性分析
            if i > 0:
                prev_frame = sorted_data[i-1]
                continuity = self._analyze_temporal_continuity(prev_frame, current_frame)
                narrative_parts.append(f"\n🔄 场景连续性:")
                narrative_parts.append(continuity)
            
            narrative_parts.append("-" * 40)
        
        # 添加整体摘要
        narrative_parts.append("\n🎯 视频内容摘要")
        if sorted_data:
            total_duration = sorted_data[-1]['timestamp']
            narrative_parts.append(f"总时长: {self.format_timestamp(total_duration, 'human')}")
        narrative_parts.append(f"总场景数: {len(sorted_data)}")
        narrative_parts.append(f"分析模型: Qwen2-VL")
        narrative_parts.append("基于视觉分析的完整视频内容报告。")
        
        narrative = "\n".join(narrative_parts)
        print("时序叙事构建完成")
        return narrative
    
    def _is_significant_change(self, frame1: Dict, frame2: Dict, threshold: float = 0.7) -> bool:
        """
        判断两个帧之间是否有显著变化
        
        Args:
            frame1, frame2: 两个帧数据
            threshold: 变化阈值
            
        Returns:
            bool: 是否有显著变化
        """
        # 基于实体变化
        entities1 = set([e["name"] for e in frame1.get("entities", [])])
        entities2 = set([e["name"] for e in frame2.get("entities", [])])
        
        if not entities1 and not entities2:
            entity_change = 0
        else:
            entity_change = len(entities1.symmetric_difference(entities2)) / max(len(entities1.union(entities2)), 1)
        
        # 基于音频变化
        audio_change = frame1.get("has_audio") != frame2.get("has_audio")
        
        return entity_change > threshold or audio_change
    
    def detect_important_events(self, integrated_data: List[Dict]) -> List[Dict]:
        """
        检测视频中的重要事件节点
        
        Args:
            integrated_data: 融合后的数据
            
        Returns:
            List[Dict]: 事件列表
        """
        print("检测重要事件...")
        
        events = []
        sorted_data = sorted(integrated_data, key=lambda x: x["timestamp"])
        
        for i in range(1, len(sorted_data)):
            current = sorted_data[i]
            previous = sorted_data[i-1]
            
            # 检测显著变化
            if self._is_significant_change(previous, current):
                event = {
                    "event_id": len(events) + 1,
                    "start_time": previous["timestamp"],
                    "end_time": current["timestamp"],
                    "start_timestamp": previous["timestamp_str"],
                    "end_timestamp": current["timestamp_str"],
                    "description": f"从 {previous['timestamp_str']} 到 {current['timestamp_str']} 的重要场景变化",
                    "importance_score": 0.8,
                    "involved_entities": list(set(
                        [e["name"] for e in previous.get("entities", [])] +
                        [e["name"] for e in current.get("entities", [])]
                    ))
                }
                events.append(event)
        
        # 按重要性排序
        events_sorted = sorted(events, key=lambda x: x["importance_score"], reverse=True)
        print(f"事件检测完成: 共检测到 {len(events_sorted)} 个重要事件")
        return events_sorted
    
    def build_entity_timelines(self, integrated_data: List[Dict]) -> Dict:
        """
        构建每个实体的出现时间线
        
        Args:
            integrated_data: 融合后的数据
            
        Returns:
            Dict: 实体时间线字典
        """
        print("构建实体时间线...")
        
        entity_timelines = {}
        
        for data in integrated_data:
            for entity in data.get("entities", []):
                entity_name = entity["name"]
                if entity_name not in entity_timelines:
                    entity_timelines[entity_name] = []
                
                entity_timelines[entity_name].append({
                    "timestamp": data["timestamp"],
                    "timestamp_str": data["timestamp_str"],
                    "action": entity.get("action", ""),
                    "description": entity.get("description", ""),
                    "type": entity.get("type", ""),
                    "context": data["combined_description"][:200] + "..."
                })
        
        print(f"实体时间线构建完成: 共追踪 {len(entity_timelines)} 个实体")
        return entity_timelines
    
    def prepare_graphrag_input(self, temporal_narrative: str, events: List[Dict], 
                              integrated_data: List[Dict]) -> str:
        """
        准备最终输入GraphRAG的文本
        
        Args:
            temporal_narrative: 时序叙事文本
            events: 事件列表
            integrated_data: 融合后的数据
            
        Returns:
            str: 最终输入GraphRAG的文本
        """
        print("准备GraphRAG输入文本...")
        
        graphrag_input = []
        
        # 1. 整体视频元数据
        graphrag_input.append("视频内容知识图谱 - Qwen2-VL分析")
        graphrag_input.append("=" * 50)
        
        # 2. 时间线叙事（主要内容）
        graphrag_input.append(temporal_narrative)
        
        # 3. 重要事件索引
        graphrag_input.append("\n\n重要事件索引")
        graphrag_input.append("=" * 30)
        for event in events[:10]:  # 前10个最重要事件
            graphrag_input.append(
                f"事件 {event['event_id']}: {event['start_timestamp']} - "
                f"{event['end_timestamp']} | {event['description'][:100]}..."
            )
        
        # 4. 实体时间线
        graphrag_input.append("\n\n实体时间线追踪")
        graphrag_input.append("=" * 30)
        entity_timelines = self.build_entity_timelines(integrated_data)
        for entity, timeline in entity_timelines.items():
            graphrag_input.append(f"{entity} ({timeline[0].get('type', '未知')}): 出现在 {len(timeline)} 个时间点")
        
        # 5. 时间关系说明
        graphrag_input.append("""
时间关系说明：
- occurs_at: 实体在特定时间点出现
- occurs_between: 事件在时间区间发生  
- precedes: 时间上的先后关系
- follows: 时间上的跟随关系
- simultaneous: 同时发生的事件
        """.strip())
        
        final_input = "\n".join(graphrag_input)
        print("GraphRAG输入文本准备完成")
        return final_input
    
    def _save_metadata(self):
        """保存元数据到文件"""
        metadata_file = self.metadata_dir / "video_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"元数据已保存: {metadata_file}")
    
    def process_video_from_frames(self, 
                                frames_data: List[Dict], 
                                audio_data: Optional[Dict] = None,
                                video_info: Optional[Dict] = None):
        """
        从提供的帧数据生成知识图谱
        
        Args:
            frames_data: 帧数据列表
            audio_data: 可选的音频数据
            video_info: 可选的视频信息
        """
        print("🎬 开始从帧数据生成知识图谱...")
        print("使用Qwen2-VL进行视觉内容分析...")
        
        # 存储视频信息
        self.metadata["video_info"] = video_info or {
            "source": "provided_frames",
            "total_frames": len(frames_data),
            "analysis_model": "Qwen2-VL"
        }
        
        # === 阶段1: 帧数据处理 ===
        print("\n=== 阶段1: 帧数据处理 ===")
        processed_frames = self.process_frames(frames_data)
        extracted_audio_data = self.extract_audio_and_subtitles(audio_data)
        
        # === 阶段2: 多模态分析 ===
        print("\n=== 阶段2: 多模态分析 ===")
        visual_data = self.analyze_visual_content(processed_frames)
        integrated_data = self.integrate_multimodal_data(visual_data, extracted_audio_data)
        
        # === 阶段3: 时序构建 ===
        print("\n=== 阶段3: 时序关系构建 ===")
        temporal_narrative = self.build_temporal_narrative(integrated_data)
        events = self.detect_important_events(integrated_data)
        
        # === 阶段4: GraphRAG集成 ===
        print("\n=== 阶段4: 知识图谱生成 ===")
        graphrag_input = self.prepare_graphrag_input(temporal_narrative, events, integrated_data)
        self.graphrag.insert(graphrag_input)
        
        # 更新并保存元数据
        self.metadata.update({
            "events": events,
            "integrated_data": integrated_data,
            "entity_timelines": self.build_entity_timelines(integrated_data),
            "video_info": {
                **self.metadata["video_info"],
                "total_frames_processed": len(integrated_data),
                "total_events": len(events),
                "total_entities": len(self.build_entity_timelines(integrated_data))
            }
        })
        
        self._save_metadata()
        print("✅ 从帧数据生成知识图谱完成！")
    


##################考虑后决定是否要留######################
    def query_video(self, question: str, use_temporal: bool = True, method: str = "local"):
        """
        查询视频内容
        
        Args:
            question: 查询问题
            use_temporal: 是否使用时序信息增强
            method: 查询方法 - "local" 或 "global"
            
        Returns:
            str: 查询结果
        """
        print(f"执行查询: {question}")
        
        # 基础查询
        response = self.graphrag.query(question, param=QueryParam(mode=method))
        
        # 时序信息增强
        if use_temporal and self.metadata.get("events"):
            temporal_context = f"\n\n📅 视频包含 {len(self.metadata['events'])} 个重要事件，"
            if self.metadata['video_info'].get('total_frames_processed'):
                temporal_context += f"共分析 {self.metadata['video_info']['total_frames_processed']} 个场景"
            response += temporal_context
        
        return response
    
    def get_video_info(self) -> Dict:
        """
        获取视频处理信息
        
        Returns:
            Dict: 视频信息
        """
        return self.metadata.get("video_info", {})
    
    def get_events(self) -> List[Dict]:
        """
        获取检测到的重要事件
        
        Returns:
            List[Dict]: 事件列表
        """
        return self.metadata.get("events", [])
    
    def get_entity_timelines(self) -> Dict:
        """
        获取实体时间线
        
        Returns:
            Dict: 实体时间线
        """
        return self.metadata.get("entity_timelines", {})
    
    def get_processing_summary(self) -> Dict:
        """
        获取处理摘要
        
        Returns:
            Dict: 处理摘要信息
        """
        video_info = self.get_video_info()
        events = self.get_events()
        entities = self.get_entity_timelines()
        
        return {
            "total_frames": video_info.get('total_frames_processed', 0),
            "total_events": len(events),
            "total_entities": len(entities),
            "analysis_model": video_info.get('analysis_model', '未知'),
            "working_directory": str(self.working_dir)
        }


# 使用示例
def main():
    """使用示例"""
    # 初始化视频处理器
    video_processor = VideoToGraphRAG(working_dir="./my_video_kg")
    
    # 示例帧数据（替换为实际的帧文件路径）
    example_frames = [
        {
            "frame_id": 0,
            "timestamp": 0.0,
            "file_path": "./frame_000001.jpg",  # 替换为实际路径
            "extraction_reason": "first_frame"
        },
        {
            "frame_id": 1,
            "timestamp": 5.0,
            "file_path": "./frame_000002.jpg",  # 替换为实际路径
            "extraction_reason": "interval"
        },
        {
            "frame_id": 2,
            "timestamp": 10.0,
            "file_path": "./frame_000003.jpg",  # 替换为实际路径
            "extraction_reason": "interval"
        }
    ]
    
    try:
        # 从帧数据生成知识图谱
        video_processor.process_video_from_frames(frames_data=example_frames)
        
        # 显示处理结果摘要
        summary = video_processor.get_processing_summary()
        print(f"\n📊 处理结果摘要:")
        print(f"- 处理帧数: {summary['total_frames']}")
        print(f"- 检测事件: {summary['total_events']}")
        print(f"- 追踪实体: {summary['total_entities']}")
        print(f"- 分析模型: {summary['analysis_model']}")
        print(f"- 工作目录: {summary['working_directory']}")
        
        # 示例查询
        queries = [
            "视频中的主要场景有哪些？",
            "找出所有重要的事件",
            "描述视频的时间线结构",
            "视频中出现了哪些主要实体？"
        ]
        
        print(f"\n🔍 示例查询:")
        for query in queries:
            print(f"\n❓ 问题: {query}")
            answer = video_processor.query_video(query, method="local")
            print(f"📝 回答: {answer}")
            
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
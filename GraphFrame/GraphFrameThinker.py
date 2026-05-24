from GraphFrame.InternVideoEncoder import InternVideoEncoder
from GraphFrame.DynamicVideoGraph import DynamicVideoGraph
from GraphFrame.Vision2GraphParser import Vision2GraphParser
from GraphFrame.InternVideoGuidedSampling import InternVideoGuidedSampling
import torch, re, argparse, decord, json
import torch.multiprocessing as mp
from Vgent.models.utils import *
import networkx as nx
import numpy as np
import pandas as pd
import shutil
import cv2
from itertools import combinations 
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoModel, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig



CONFIG = {
    "MODEL_PATH": "/home/teacher_tang/Video-Holmes/FramerThinker_series/Qwen2.5-VL-7B-Instruct",
    "BASE_VIDEO_DIR": "/home/teacher_tang/Video-Holmes/InternVL_2_5_HiCo_R64_series/Benchmark/videos_cropped",
    "BASE_FRAME_DIR_ROOT": "/home/teacher_tang/Video-Holmes/GraphFrame/temporary_framer",
    "TARGET_JSON_PATH": "/home/teacher_tang/Video-Holmes/InternVL_2_5_HiCo_R64_series/Benchmark/test_Video-Holmes.json",
    "MAX_ITERATIONS": 5,
    "MAX_RETRIES": 3,
    "NUM_FRAMES_TO_SAMPLE": 8,
    "NUM_FRAMES_TO_SAMPLE_LONG": 12,
    "DEFAULT_GPUS": 2, 
    "MAX_FRAME_WIDTH": 640,
    "MAX_FRAME_HEIGHT": 360,
    "MAX_FRAME_WIDTH_LONG": 448,
    "MAX_FRAME_HEIGHT_LONG": 252,
    "EXCEL_LOG_PATH": "/home/teacher_tang/Video-Holmes/GraphFrame/results/evaluation_details_new.xlsx",
    "RESULTS_JSONL_PATH": "/home/teacher_tang/Video-Holmes/GraphFrame/results/results_metrics_new.jsonl"
}

class GraphFrameThinker:

    def __init__(self, device, model_name="/home/teacher_tang/Video-Holmes/FramerThinker_series/Qwen2.5-VL-7B-Instruct"):
       
        self.video_cache = {}
        self.sampler = None
        self.current_video_path = None    ##########要在全局改动###########
        self.embedding_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        self.embedding_model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto", 
            trust_remote_code=True
        ).to(device)
        self.config = AutoConfig.from_pretrained("/home/teacher_tang/Video-Holmes/InternVL_2_5_HiCo_R64_series/InternVL_2_5_HiCo_R64", trust_remote_code=True, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(
                    "/home/teacher_tang/Video-Holmes/InternVL_2_5_HiCo_R64_series/InternVL_2_5_HiCo_R64",
                    config=self.config,
                    trust_remote_code=True,
                    dtype=torch.bfloat16,
                    local_files_only=True
                ).to(device)
        self.video_encoder = InternVideoEncoder(self.config, self.encoder)
    
    def run_inference(self, model, processor, messages, device):

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=0.6, top_p=0.9)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = \
        processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(output_text)

        return output_text

    def parse_model_response(self, response):
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        action_match = re.search(r"<action>(.*?)</action>", response, re.DOTALL)
        return (think_match.group(1).strip(), action_match.group(1).strip()) if think_match and action_match else (
        None, None)
    
    def get_video_metadata(self, video_path):
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
            frame_count = len(vr)
            del vr
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps, frame_count
        except Exception:
            return 0, 0
        
    def scale_down_preserving_aspect_ratio(self, image, max_width=640, max_height=360):
        h, w = image.shape[:2]

        if w <= max_width and h <= max_height:
            return image

        ratio_w = max_width / w
        ratio_h = max_height / h
        scale_ratio = min(ratio_w, ratio_h)

        new_w = int(w * scale_ratio)
        new_h = int(h * scale_ratio)

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return resized_image
    
    def extract_frames(self, video_path, frame_indices, output_dir, max_width, max_height):
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
            frames_array = vr.get_batch(frame_indices).asnumpy()
            for i, frame_idx in enumerate(frame_indices):
                frame_img_bgr = cv2.cvtColor(frames_array[i], cv2.COLOR_RGB2BGR)
                scaled_frame = self.scale_down_preserving_aspect_ratio(
                    frame_img_bgr,
                    max_width=max_width,
                    max_height=max_height
                )
                output_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
                cv2.imwrite(output_path, scaled_frame)
                saved_paths.append(output_path)
            del vr
        except Exception as e:
            print(f"!!!!!! AN EXCEPTION OCCURRED in extract_frames !!!!!!")
            print(f"Video Path: {video_path}")
            print(f"Frame Indices: {frame_indices}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
        return saved_paths

    def parse_to_frame(self, value, fps):
        if ":" in value:
            parts = value.split(":")
            # 处理 MM:SS 或 HH:MM:SS
            if len(parts) == 2:
                seconds = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            return int(seconds * fps)
        else:
            return int(value)

    def check_looping_behavior_choose(self, current_action, action_history, fps):
        """
        检查当前动作是否在原地打转
        """
        # match = re.search(r"choose frames between\s+(\d+)\s+and\s+(\d+)", current_action.strip())
        match = re.search(r"choose frames between\s+([\d:]+)\s+and\s+([\d:]+)", current_action.strip()) 
        if not match:
            return False, None
            
        try:
            s_raw, e_raw = match.groups()
            s, e = sorted([self.parse_to_frame(s_raw, fps), self.parse_to_frame(e_raw, fps)])
            current_range = (s, e)
        except:
            return False, None

        for prev_range in action_history:
            # 完全相同或高度重合（重合度超过90%）
            overlap_s = max(current_range[0], prev_range[0])
            overlap_e = min(current_range[1], prev_range[1])
            if overlap_e > overlap_s:
                overlap_len = overlap_e - overlap_s
                max_len = max(current_range[1]-current_range[0], prev_range[1]-prev_range[0])
                if overlap_len / max_len > 0.95:
                    return True, current_range
                    
        return False, current_range

    def handle_choose_frames(self, action_content, video_path, video_id, iteration, total_frames, base_frame_dir,
                         num_frames_to_sample, max_width, max_height, fps):
        # match = re.search(r"choose frames between\s+(\d+)\s+and\s+(\d+)", action_content.strip())
        
        match = re.search(r"choose frames between\s+([\d:]+)\s+and\s+([\d:]+)", action_content.strip()) 
        if not match: return None, []
        try: 
            raw_start, raw_end = match.groups()
            start_frame = self.parse_to_frame(raw_start, fps)
            end_frame = self.parse_to_frame(raw_end, fps)
        except ValueError:
            return None, []
        if start_frame >= end_frame or end_frame >= total_frames or (end_frame - start_frame) <= num_frames_to_sample:
            return None, []
        frame_indices = np.linspace(start_frame , end_frame, num_frames_to_sample, dtype=int).tolist()
        _frame_indices = []
        for i in frame_indices:
            _frame_indices.append(int(round(i)))
        output_dir = os.path.join(base_frame_dir, f"{video_id}_iter_{iteration}")
        image_paths = self.extract_frames(video_path, frame_indices, output_dir, max_width, max_height)
        user_content = []
        for path, index in zip(image_paths, frame_indices):
            user_content.extend([{"type": "text", "text": f"frame {index}:"}, {"type": "image", "image": path}])
        return user_content, _frame_indices

    def handle_get_frame_number(self, action_content, video_path, video_id, iteration, total_frames, base_frame_dir,
                         num_frames_to_sample, max_width, max_height, fps):
        time_match = re.match(r'get frame number at time\s+([\d:]+)', action_content.strip())
        if not time_match: return None, []
        try:
            t = time_match.group(1)
        except ValueError:
            return None, []
        frame = self.parse_to_frame(t, fps)
        if frame < 0 or frame >= total_frames:
            return None, []
        
        step = max(1, round((fps * 3) / num_frames_to_sample))
        half_width = (num_frames_to_sample * step) // 2
        start = frame - half_width
        end = start + (num_frames_to_sample * step) 
        if start < 0:
            offset = -start
            start += offset
            end += offset
        if end > total_frames:
            offset = end - total_frames
            start -= offset
            end -= offset
        start = max(0, start)
        frame_indices = np.linspace(start, end - 1, num_frames_to_sample-1, dtype=int).tolist()
        frame_indices.append(frame)
        frame_indices = list(set(frame_indices))
        _frame_indices = []
        for i in frame_indices:
            _frame_indices.append(int(round(i)))
        _frame_indices.sort()
        output_dir = os.path.join(base_frame_dir, f"{video_id}_iter_{iteration}")
        image_paths = self.extract_frames(video_path, frame_indices, output_dir, max_width, max_height)
        user_content = []
        for path, index in zip(image_paths, frame_indices):
            user_content.extend([{"type": "text", "text": f"frame {index}:"}, {"type": "image", "image": path}])
        return user_content, _frame_indices

    def stringify_conversation(self, conversation_history: list, fps):

        frame_intervals = []
        
        for turn in conversation_history:
            if turn['role'] == 'assistant':
                content = turn['content']
                # cf_match = re.search(r"choose frames between\s+(\d+)\s+and\s+(\d+)", content)
                cf_match = re.search(r"choose frames between\s+([\d:]+)\s+and\s+([\d:]+)", content) 
                if cf_match:
                    s_raw, e_raw = cf_match.groups()
                    s, e = sorted([self.parse_to_frame(s_raw, fps), self.parse_to_frame(e_raw, fps)])
                    frame_intervals.append((s, e))

                gf_match = re.search(r"get frame number at time\s+([\d:]+)", content)
                if gf_match:
                    t_raw = gf_match.group(1)
                    t = self.parse_to_frame(t_raw, fps)
                    frame_intervals.append((t,))

        if len(frame_intervals) > 1:
            for interval1, interval2 in combinations(frame_intervals, 2):
                if len(interval1) == 2 and len(interval2) == 2:
                    s1, e1 = interval1
                    s2, e2 = interval2
                    if s1 == s2 and e1 == e2:
                        return "error"
                    if abs(s1 - s2) <= 1 and abs(e1 - e2) <= 1:
                        return "error"
                    
                if len(interval1) == 1 and len(interval2) == 1:
                    s1 = interval1[0]
                    s2 = interval2[0]
                    if s1 == s2 :
                        return "error"
                    if abs(s1 - s2) <= 1:
                        return "error"

        stringified_content = ""
        for turn in conversation_history[1:]:
            content = turn['content']
            if turn['role'] == 'user' or turn['role'] == 'assistant':
                content = content.strip()
                stringified_content += f"\n{turn['role'].upper()}: {content}\n"
            
        return stringified_content
    
    def validate_reasoning_process(self, predict_str: str, num_frames_to_sample: int, fps):

        try:
            think_contents = re.findall(r'<think>(.*?)</think>', predict_str, re.DOTALL)
            action_contents = re.findall(r'<action>(.*?)</action>', predict_str, re.DOTALL)
        except Exception:
            return False, 0, 0

        # 必须有内容，且 think 和 action 必须成对出现
        if not think_contents or not action_contents or len(think_contents) != len(action_contents):
            return False, 0, 0
        
        # 终止条件校验：最后一轮动作必须是输出答案
        if not action_contents[-1].strip().startswith('output answer:'):
            return False, 0, 0

        # tool_call_count: 模型尝试深入分析的次数（不含最后一次输出答案）
        tool_call_count = len(action_contents) - 1
        summary_add_count = 0  # 摘要细化次数
        action_time_history = []
        action_frame_pairs = []

        # 5. 逐轮校验中间动作
        for action in action_contents[:-1]:
            action = action.strip()

            # 校验 choose frames 动作格式
            cf_match = re.match(r'choose frames between ([\d:]+) and ([\d:]+)', action)
            # 校验 get frame number 动作格式
            gf_match = re.match(r'get frame number at time ([\d:]+)', action)
            if gf_match:
                summary_add_count += 1
                t = gf_match.group(1)

                if t in action_time_history:
                    return False, 0, 0
                action_time_history.append(t)
                continue 

            if cf_match:
                summary_add_count += 1
                raw_s, raw_e = cf_match.groups()
                num1 = self.parse_to_frame(raw_s, fps)
                num2 = self.parse_to_frame(raw_e, fps)
                start_f, end_f = min(num1, num2), max(num1, num2)
                current_pair = (start_f, end_f)
                if current_pair in action_frame_pairs: 
                    return False, 0, 0
                action_frame_pairs.append(current_pair)
                if end_f - start_f < num_frames_to_sample: 
                    return False, 0, 0
                
                continue
            return False, 0, 0
        return True, tool_call_count, summary_add_count

    def solve(self, qa_item, device, rank, threshold_seconds=1.0, max_turns=6, subtitles=None):
        """
        完整的推理流程 
        """
        # ===== Stage 0: 全局编码 =====
        print("🎬 Stage 0: Encoding video with InternVideo2.5...")
        video_id, question, correct_answer, candidates = qa_item["video"], qa_item['Question'], qa_item['Answer'], qa_item['Options']
        video_path = os.path.join(CONFIG["BASE_VIDEO_DIR"], f"{video_id}.mp4")

        video_fps, total_frames = self.get_video_metadata(video_path)
        if total_frames == 0 or video_fps == 0: return {"status": "format_error"}, None
        duration_in_seconds = total_frames / video_fps
        if duration_in_seconds > 300:
            num_frames_to_sample = CONFIG["NUM_FRAMES_TO_SAMPLE_LONG"]
            max_frame_width = CONFIG["MAX_FRAME_WIDTH_LONG"]
            max_frame_height = CONFIG["MAX_FRAME_HEIGHT_LONG"]
        else:
            num_frames_to_sample = CONFIG["NUM_FRAMES_TO_SAMPLE"]
            max_frame_width = CONFIG["MAX_FRAME_WIDTH"]
            max_frame_height = CONFIG["MAX_FRAME_HEIGHT"]

        video_features, fps_1_frame_indices, = self.video_encoder.encode_full_video(video_path)
        importance_scores = self.video_encoder.get_temporal_importance(video_features)
        self.sampler = InternVideoGuidedSampling(self.encoder, video_features, importance_scores)
        self.v2g_parser = Vision2GraphParser(device, self.vlm, self.processor)
        # print(f"\ninitial_graph_context {initial_graph_context}")
        total_seconds = total_frames / video_fps
        max_time_str = f"{int(total_seconds // 60):02d}:{int(total_seconds % 60):02d}"
    
        # ===== Stage 1: 初始观察 =====
        print("🔍 Stage 1: Initial observation...")

        total_seconds = total_frames / video_fps
        max_time_str = f"{int(total_seconds // 60):02d}:{int(total_seconds % 60):02d}"

        system_prompt = f"""
        You are an expert AI assistant that answers questions about a video by iteratively analyzing the graph information generated by it.
        Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.

        [ACTION CATEGORIES]
        1. `choose frames between START_FRAME and END_FRAME`: Request a more detailed view of a specific video segment. Both START_FRAME_INDEX and END_FRAME_INDEX is frame index. 
        2. `get frame number at time MM:SS`: Tool for time-to-frame conversion. CONSTRAINT: 00:00 <= MM:SS <= {max_time_str}.
        3. `output answer: OPTION`: Final step ONLY. The OPTION MUST be a single letter from {list(candidates.keys())}.

        [TECHNICAL METADATA]
        - Video FPS: {video_fps} | Total Frames: {total_frames} | Duration: {max_time_str}.
        - Formula: Frame = seconds * {video_fps}. (e.g., 0:28 = 28 * {video_fps} = {int(28 * video_fps)}).
        - Constraints: 0 <= START_FRAME < END_FRAME <= {total_frames-1} AND (END_FRAME - START_FRAME) >= {num_frames_to_sample}.

        [NOTE]
        - You can use keywords in QUESTION and CANDIDATE OPTIONS or content in INFERENCE FLOW to guide next action.
        - If your current action is `choose frames between START_FRAME and END_FRAME`, You must select a frame (integer) range or a MM:SS range that are different from those appearing in INFERENCE FLOW.
        - IF your current action is `get frame number at time MM:SS`, you must select a MM:SS that are different from those appearing in INFERENCE FLOW.
        - Base your reasoning on subtle cues.
        - No guessing.
        - Your display MUST reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag. (i.e., <think> your reasoning </think> <action> your action </action>)
        """

        base_prompt_text = f" QUESTION:{question} CANDIDATE OPTIONS:{candidates}"
        base_frame_dir = os.path.join(CONFIG["BASE_FRAME_DIR_ROOT"], f"gpu_{rank}")

        # ===== Stage 2: 多轮推理 ===== 
        trajectory = []

        graph = nx.MultiDiGraph() 
        self.graph_builder = DynamicVideoGraph(graph, self.embedding_tokenizer, self.embedding_model)
        
        for times in range(CONFIG["MAX_RETRIES"]):
            if times > 1:
                system_prompt = f"""You are an expert AI assistant that answers questions about a video by iteratively analyzing provided visual summaries.
        Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.

        Actions MUST be: `output answer: OPTION`: Provide the final answer(e.g., A, B, C...). DO NOT output "None of the above" or any option not in the candidates. You MUST pick the most likely one from the list and you MUST pick one.

        [CRITICAL CONSTRAINTS]
        - Your response MUST begin with <think> and end with </action>.
        - The final answer MUST be one of the provided candidates ({candidates}). 
        - Only output `output answer: OPTION`.
        - Current video FPS: {video_fps}. 
        - Use the formula: frame = seconds * {video_fps} to convert candidate timestamps to frame numbers.
        """
   
            reasoning_memory = [] # 记录每一轮的 <think> 内容
            action_history = []   # 记录历史选帧范围，用于死循环检测
            error_msg = None

            if os.path.exists(base_frame_dir): shutil.rmtree(base_frame_dir)
            os.makedirs(base_frame_dir)

            initial_frame_indices = self.sampler.suggest_initial_frames(fps_1_frame_indices=fps_1_frame_indices, num_frames=num_frames_to_sample)
            _initial_frame_indices = []
            for i in initial_frame_indices:
                _initial_frame_indices.append(int(round(i)))

            initial_frames_dir = os.path.join(base_frame_dir, f"{video_id}")
            initial_image_paths = self.extract_frames(video_path, _initial_frame_indices, initial_frames_dir, max_frame_width,
                                             max_frame_height)
            
            if not initial_image_paths:
                print(f"Warning: Initial frame extraction failed for {video_id}. Retrying...")
                continue
            initial_user_content = [{"type": "text", "text": "Extract critical Entities, Relations, Description and Scene"}]
            for path, index in zip(initial_image_paths, _initial_frame_indices):
                initial_user_content.extend([{"type": "text", "text": f"frame {index}:"}, {"type": "image", "image": path}])

            initial_subgraph = self.v2g_parser.parse(
                frame_indices=initial_frame_indices,
                video_fps=video_fps,
                user_content=initial_user_content
            )
            self.graph_builder.add_subgraph(initial_subgraph, turn_id=0)
            initial_graph_context = self.query_graph(question, candidates, subtitles, threshold_seconds, 0)
            prompt_text = base_prompt_text + f"Current summary(information about video):{initial_graph_context}"

            final_answer = None
            conversation_history = [{"role": "system", "content": system_prompt}]
            clean_conversation_history = [{"role": "system", "content": system_prompt}]
            selected_frames = set(_initial_frame_indices)

            trajectory.append({
            'times': times,
            'turn': 0,
            'think': None,
            'action': None,
            'observation': f"Observed frames {_initial_frame_indices}",
            'subgraph': initial_subgraph
        })

            for turn in range(1, max_turns):
                print(f"times {times} turn {turn} infer")

                clean_conversation_history.append({"role": "user", "content": prompt_text})
                full_prompt_for_inference = prompt_text + f" Frames that have been selected {selected_frames}"

                if reasoning_memory:
                    memory_summary = "\n".join(reasoning_memory)
                    full_prompt_for_inference += f" INFERENCE FLOW: {memory_summary}"
                if error_msg:
                    full_prompt_for_inference += f" {error_msg}"
                    error_msg = None

                message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt_for_inference} 
                ]
                # 记录带上下文的对话历史（用于 debug 或模型连续对话）
                conversation_history.append(message[1])

                model_response_str = self.run_inference(self.vlm, self.processor, message, device)
                conversation_history.append({"role": "assistant", "content": model_response_str})
                clean_conversation_history.append({"role": "assistant", "content": model_response_str})
                think, action = self.parse_model_response(model_response_str)
                print(f"\nAction: {action}")

                if action is None: break

                infer_flow = f"time {times} turn {turn} inference flow: {model_response_str}"
                print(infer_flow)
                reasoning_memory.append(infer_flow)

                # if "choose frames" in action:
                #     is_looping, current_range = self.check_looping_behavior(action, action_history, video_fps)
                #     if is_looping:
                #         error_msg = f'''ERROR: You are repeating the same frame range {action} which provided no new info. 
                #             DO NOT request this range again. Analyze the keywords in Candidates 
                #             and search for DIFFERENT timestamps or objects to break the loop.'''
                #         continue 
                #     if current_range:
                #         action_history.append(current_range)

                if action.startswith("output answer:"):
                    final_answer = action.replace("output answer:", "").strip()
                    if final_answer not in list(candidates.keys()):
                        error_msg = f'''ERROR: '{final_answer}' is not a valid candidate. You MUST choose one from {list(candidates.keys())}.
                            If you cannot find the answer, you MUST use 'choose frames' to explore the video further.
                            Search near the time or for the entities mentioned in the question.'''
                        continue
                    else:
                        print(f"\nFinal Answer: {final_answer}")
                        break

                if "choose frames" in action:
                    user_content, frame_indices = self.handle_choose_frames(
                        action, 
                        video_path, 
                        video_id, 
                        turn, 
                        total_frames, 
                        base_frame_dir, 
                        num_frames_to_sample, 
                        max_frame_width, 
                        max_frame_height, 
                        video_fps
                        )
                    if frame_indices:
                        selected_frames.update(frame_indices)
                        new_subgraph = self.v2g_parser.parse(
                        frame_indices=frame_indices,
                        video_fps=video_fps,
                        user_content=user_content
                    )
                        self.graph_builder.add_subgraph(new_subgraph, turn_id=turn)
                        observation = f"Observed frames {frame_indices}" if frame_indices else None
                        trajectory.append({
                            'times': times,
                            'turn': turn,
                            'think': think,
                            'action': action,
                            'observation': observation,
                            'subgraph': new_subgraph
                        })
                        graph_context = self.query_graph(question, candidates, subtitles, threshold_seconds, turn)
                        prompt_text = base_prompt_text + f"Current summary(information about video):{graph_context}"
                    else:
                        prompt_text = base_prompt_text 

                if "get frame" in action:
                    user_content, frame_indices = self.handle_get_frame_number(
                        action, 
                        video_path, 
                        video_id, 
                        turn, 
                        total_frames, 
                        base_frame_dir, 
                        num_frames_to_sample, 
                        max_frame_width, 
                        max_frame_height, 
                        video_fps
                        )
                    if frame_indices:
                        selected_frames.update(frame_indices)
                        new_subgraph = self.v2g_parser.parse(
                        frame_indices=frame_indices,
                        video_fps=video_fps,
                        user_content=user_content
                    )
                        self.graph_builder.add_subgraph(new_subgraph, turn_id=turn)
                        observation = f"Observed frames {frame_indices}" if frame_indices else None
                        trajectory.append({
                            'times': times,
                            'turn': turn,
                            'think': think,
                            'action': action,
                            'observation': observation,
                            'subgraph': new_subgraph
                        })
                        graph_context = self.query_graph(question, candidates, subtitles, threshold_seconds, turn)
                        prompt_text = base_prompt_text + f"Current summary(information about video):{graph_context}"
                    else:
                        prompt_text = base_prompt_text 

            if final_answer is not None:
                # print(f"clean_conversation_history {clean_conversation_history}")
                full_str = self.stringify_conversation(clean_conversation_history, video_fps)
                is_valid, tool_calls, summary_adds = self.validate_reasoning_process(full_str, num_frames_to_sample, video_fps)
                if is_valid:
                    images_used = num_frames_to_sample * (1 + summary_adds)
                    status = "correct" if final_answer == correct_answer else "wrong_answer"
                    return {"status": status, "tool_calls": tool_calls, "images_used": images_used}, final_answer
    
        return {"status": "format_error"}, final_answer
    
    def query_graph(self, question, candidates, subtitles, threshold_seconds, turn):
        """
        从图谱中查询相关信息
        """
        # 1. 获取图谱摘要
        graph_summary = self.graph_builder.summarize()
        
        # 2. 查询相关证据链
        evidence_chains = self.graph_builder.query_evidence_chains(question, candidates, subtitles)
        
        # 3. 发现可疑模式（仅在悬疑推理任务）
        suspicious_patterns = []
        if "who" in question.lower() or "why" in question.lower():
            suspicious_patterns = self.graph_builder.find_suspicious_patterns(threshold_seconds) # threshold_seconds用于空间矛盾检测
        
        # # 4. 时序线索
        # temporal_clues = self.graph_builder.get_temporal_sequence()  # 是想干什么，怎么写入组合上下文
        
        # 组合上下文
        context = f"""
        === Graph Knowledge Base (Turn {turn}) ===
        
        Entities: {graph_summary['entities']}
        Relations: {graph_summary['relations']}
        Temporal Chain: {graph_summary['temporal_chain']}
        
        Evidence Chains:
        {evidence_chains}
        
        Suspicious Patterns:
        {suspicious_patterns}
        """
        
        return context
        
def aggregate_and_print_results(all_results):
        total_problems = len(all_results)
        if total_problems == 0:
            print("No results to aggregate.")
            return

        stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        total_correct = 0
        total_wrong_answer = 0
        total_format_error = 0
        total_images_used = 0

        for res in all_results:
            status = res.get("status") 
            if status == "correct":
                total_correct += 1
                tool_calls = res.get("tool_calls", 0)
                stats[tool_calls]['correct'] += 1
                stats[tool_calls]['total'] += 1
                total_images_used += res.get("images_used", 0)
            elif status == "wrong_answer":
                total_wrong_answer += 1
                tool_calls = res.get("tool_calls", 0)
                stats[tool_calls]['total'] += 1
                total_images_used += res.get("images_used", 0)
            else:
                total_format_error += 1

        print("\n" + "=" * 50)
        print(" " * 15 + "AGGREGATED RESULTS")
        print("=" * 50)
        print(f"Total Problems Processed: {total_problems}\n")

        print("--- Overall Performance ---")
        print(f"Correct Answers: {total_correct} ({total_correct / total_problems:.2%})")
        print(f"Wrong Answers (Valid Format): {total_wrong_answer} ({total_wrong_answer / total_problems:.2%})")
        print(f"Format Errors (After Retries): {total_format_error} ({total_format_error / total_problems:.2%})")
        print("-" * 25)

        valid_format_problems = total_correct + total_wrong_answer
        if valid_format_problems > 0:
            avg_images = total_images_used / valid_format_problems
            print(f"Average Images Used (per valid problem): {avg_images:.2f}")

            total_tool_calls_made = 0
            for i in stats.keys():
                total_tool_calls_made += i * stats[i]['total']

            avg_tool_calls = total_tool_calls_made / valid_format_problems
            print(f"Average Tool Calls Used (per valid problem): {avg_tool_calls:.2f}")

        print("\n--- Tool Call Analysis (for valid format attempts) ---")
        sorted_tool_calls = sorted(stats.keys())
        for i in sorted_tool_calls:
            correct = stats[i]['correct']
            total = stats[i]['total']
            accuracy = correct / total if total > 0 else 0
            print(f"Problems with {i} tool calls: {total} | Accuracy: {accuracy:.2%}")

        print("=" * 50)

def worker(rank, world_size, data_chunks, results_list):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = torch.device(f"cuda:{rank}")
    data_chunk = data_chunks[rank]
    thinker = GraphFrameThinker()

    local_results = []
    for i, qa_item in enumerate(data_chunk):
        print(f"[GPU {rank}] Processing item {i + 1}/{len(data_chunk)}...")
        result = thinker.solve(qa_item, 8, device)
        local_results.append(result)

    results_list.extend(local_results) 
    
# def main():
#     parser = argparse.ArgumentParser(description="Run multi-GPU video QA evaluation.")
#     parser.add_argument("-n", "--num_gpus", type=int, default=CONFIG["DEFAULT_GPUS"],
#                         help=f"Number of GPUs to use. Default: {CONFIG['DEFAULT_GPUS']}")
#     args = parser.parse_args()
#     world_size = args.num_gpus

#     try:
#         with open(CONFIG["TARGET_JSON_PATH"], 'r', encoding='utf-8') as f:
#             full_dataset = json.load(f)
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         return

#     data_chunks = np.array_split(full_dataset, world_size)

#     with mp.Manager() as manager:
#         results_list = manager.list()
#         mp.spawn(worker,
#                  args=(world_size, data_chunks, results_list),
#                  nprocs=world_size,
#                  join=True)

#         final_results = list(results_list)

#     aggregate_and_print_results(final_results)
    
def main():

    # 加载数据集
    try:
        with open(CONFIG["TARGET_JSON_PATH"], 'r', encoding='utf-8') as f:
            full_dataset = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    processed_qids = set()
    if os.path.exists(CONFIG["RESULTS_JSONL_PATH"]):
        with open(CONFIG["RESULTS_JSONL_PATH"], 'r', encoding='utf-8') as f:
            for line in f:
                res = json.loads(line)
                processed_qids.add(res.get("Question ID"))

    device = torch.device("cuda:0") 
    thinker = GraphFrameThinker(device)
    results_list = []
    
    print(f"Starting single-process evaluation on {len(full_dataset)} items...")
    
    # 直接循环处理
    for i, qa_item in enumerate(full_dataset):
        qid = qa_item.get("Question ID")
        
        # 断点跳过逻辑
        if qid in processed_qids:
            continue

        print(f"Processing item {i + 1}/{len(full_dataset)}...")
        # 调用处理函数
        result, final_answer = thinker.solve(qa_item=qa_item, device="cuda", rank=0)
        result["Question ID"] = qid
        results_list.append(result)

        video_id = qa_item["video"]
        detail_data = {
            "index": i,
            "video": video_id,
            "video_type": qa_item.get("Question Type", "Unknown"),
            "question": qa_item["Question"],
            "answer": qa_item["Answer"],
            "dimensions": str(list(qa_item.get("Options", {}).keys())), # 或者根据需要记录维度
            "video_path": os.path.join(CONFIG["BASE_VIDEO_DIR"], f"{video_id}.mp4"),
            "prediction": final_answer
        }

        # 将当前行追加到 Excel
        df_new = pd.DataFrame([detail_data])
        if not os.path.exists(CONFIG["EXCEL_LOG_PATH"]):
            df_new.to_excel(CONFIG["EXCEL_LOG_PATH"], index=False)
        else:
            with pd.ExcelWriter(CONFIG["EXCEL_LOG_PATH"], mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                try:
                    start_row = writer.book['Sheet1'].max_row
                except:
                    start_row = 0
                df_new.to_excel(writer, index=False, header=False, startrow=start_row)

        with open(CONFIG["RESULTS_JSONL_PATH"], 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    final_results_to_print = []
    if os.path.exists(CONFIG["RESULTS_JSONL_PATH"]):
        with open(CONFIG["RESULTS_JSONL_PATH"], 'r', encoding='utf-8') as f:
            for line in f:
                final_results_to_print.append(json.loads(line))
    
    print("\n--- Evaluation Finished ---")
    aggregate_and_print_results(final_results_to_print)

    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# system_prompt = f"""You are an expert AI assistant that answers questions about a video by iteratively analyzing provided visual summaries.
# Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.

# Actions must come from:
# 1. `choose frames between START_FRAME and END_FRAME`: Request a more detailed visual summary of a specific video segment. The number of summarized frames is fixed, currently {num_frames_to_sample}. Example: <action>choose frames between 100 and 120</action>, must be 'choose frames'.
# 2. `output answer: OPTION`: Provide the final answer (e.g., A, B, C...) when you are confident. The question offers available candidates; the final answer MUST come from the candidates and you must choose one. Example: <action>output answer: C</action>.

# Constraints:
# - Your response MUST begin with <think> and end with </action>.
# - Format: <think> [Your reasoning] </think> <action> [Specific action] </action>
# - No other formatting is allowed.
# - 0 <= START_FRAME < END_FRAME <= {total_frames}.
# - END_FRAME - START_FRAME >= {num_frames_to_sample}.
# - Current video FPS: {video_fps}. 
# - When seconds (e.g., 30s, 40s) in candidates, you MUST know the relation between frame number and seconds: frame number = seconds × fps.
# - You can use seconds as a bridge to find the relation between frame number and MM:SS values.

# You will be provided with a text-based summary of the key information that have known currently of video. Use these descriptions to think and plan your next action.
# """
# RULE_REMINDER = f"""
# [Reminder]
# - Question:{question} Available candidates:{candidates}
# - Format: <think>...</think> <action>...</action>
# - Actions must come from:
# 1. `choose frames between START_FRAME and END_FRAME`: Use this when you are not 100% sure about the details. If the current summary does not contain the exact timestamp or clear visual evidence needed, you MUST use this action to zoom in on a specific segment. 
# 2. `output answer: OPTION`: Provide the final answer (e.g., A, B, C...) when you are confident. The question offers available candidates; the final answer MUST come from the candidates and you must choose one. Example: <action>output answer: C</action>.
# - Formula: frame = seconds * {video_fps}
# - Limits: 0 <= START_FRAME < END_FRAME <= {total_frames} (END_FRAME - START_FRAME >= {num_frames_to_sample})
# - Use current summary to think and plan your next action.
# - You have {max_turns} attempts; use them to increase your confidence.
# """

# RULE_REMINDER = f"""
# [Reminder]
# - Question:{question} Available candidates:{candidates}
# - Format: <think>Detailed analysis of current evidence</think> <action>choose frames between X and Y</action> OR <action>get frame number at time MM:SS</action> OR <action>output answer: OPTION</action>
# - The MOST IMPORTANT: DO NOT output "None of the above" or any option not in the candidates. You MUST pick the most likely one from the list and you MUST pick one.
# - Only output `output answer: OPTION` when you have verified the timestamp or action in the video segment.
# - Formula: frame = seconds * {video_fps}
# - Limits: 0 <= START_FRAME < END_FRAME <= {total_frames-1} (END_FRAME - START_FRAME >= {num_frames_to_sample})
# - Limits: 0 <= MM <=60 AND 0 <= SS <=60 AND 0 <= (MM * 60 + SS) <= {total_frames/float(video_fps)}.
# - Use current summary to think and plan your next action.
# - DETER GUESSTIMATE: If the current summary does not contain the exact timestamp or clear visual evidence needed, you MUST use `choose frames between X and Y` OR `get frame number at time MM:SS` to zoom in on a specific segment. 
# - You have {max_turns} attempts; use them to increase your confidence.
# """

#         RULE_REMINDER = f"""
# [Reminder]
# - **Format**: <think>Detailed analysis of current evidence</think> <action>choose frames between X and Y</action> OR <action>get frame number at time MM:SS</action> OR <action>output answer: OPTION</action>
# - **Candidates**: {candidates} (You MUST choose one, NO 'None of the above').
# - **Current Evidence Check**: Have you searched for the specific names, objects, or timestamps mentioned in: "{question}"?
# - **Search First**: If you haven't seen the evidence in the text summary, your action MUST be `choose frames between START_FRAME and END_FRAME`.
# - **Math Check**: 0:28s is frame {int(28 * video_fps)}. Use `get frame number at time MM:SS` if you need precision.
# - **Goal**: Find the hidden cues (like text messages, specific gestures, or micro-states) as described in the investigation protocol.
# [TECHNICAL METADATA]
#         - Video FPS: {video_fps} | Total Frames: {total_frames} | Duration: {max_time_str}.
#         - Formula: Frame = seconds * {video_fps}. (e.g., 0:28 = 28 * {video_fps} = {int(28 * video_fps)}).
#         - Constraints: 0 <= START_FRAME < END_FRAME <= {total_frames-1} AND (END_FRAME - START_FRAME) >= {num_frames_to_sample}. START_FRAME, END_FRAME MUST BE integer OR MM:SS.
#         - 00:00 <= MM:SS <= {max_time_str}
# """

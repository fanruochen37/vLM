from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import re

class Vision2GraphParser:
    def __init__(self, device, vlm, processor):

        self.processor = processor
        self.record_frame_index = set()
        self.vlm = vlm.to(device)
        self.device = device

    def mllm_response(self, messages):

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt").to(self.device)
        generated_ids = self.vlm.generate(
            **inputs, 
            max_new_tokens=4096, 
            do_sample=True, 
            temperature=0.6, 
            top_p=0.9)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = \
        self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return output_text
    
    def describe_frame(self, message):
        attempts = 0
        while attempts < 5:
            try:
                response = self.mllm_response(message)
                clean_res = response.replace("```json", "").replace("```", "").strip()
                import re
                match = re.search(r'\{.*\}', clean_res, re.DOTALL)
                if match:
                    clean_res = match.group()
                
                info = json.loads(clean_res)
                # print(info)
                entities = info.get("entities", [])
                # print(f"\n describe_frame entities {entities}")
                relations = info.get("relations", [])
                description = info.get("description", " ")
                scene = info.get("scene", " ")

                return entities, relations, description, scene
            
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                attempts += 1
        
        return [], [], " ", " "
    
    def parse(self, frame_indices, video_fps, user_content):

        GLOBEL_PEOMPT = '''
You are a forensic visual analyst, perform a panoramic structured analysis of the provided video frame. Your goal is to extract critical Entities, Relations, Description and Scene.

[Analysis Dimensions]
1. Identify all distinct entities (human, object, location, text on the video etc.). Assign unique IDs (person_A, object_cup_B). Describe materials, textures, scale, and immediate physical states (tense, static, occluded).
2. Use Subject-Verb-Object (S-V-O) for micro-actions and interactions. Identify causal traces (e.g., stains implying prior spills).
3. Do NOT report background clutter (e.g., generic trees, walls) unless they provide critical spatial context (e.g., person_A is hiding behind location_wall_B).
4. If ego-view, identify 'ego_player' and describe hand gestures/relative velocity.
5. Relation type includes:
        - left_of, inside, on_top_of, adjacent_to, etc.
        - holding, pulling, moving_towards, etc.
        - caused_by, activated_by, belongs_to -> MUST include "because" in description.
        - watching, communicating_with, etc.
[JSON Schema]
{
    "description": "A comprehensive string including: 1.People details; 2.Object states/coordinates; 3.Location/lighting context; 4.Interaction S-V-O chains; 5.Text on the video, including Subtitles, Captions, Overlay text...",
    "entities": [
        {
            "entity id": "unique_id",
            "type": "one of [human, animal, plant, location, object, naturalobject, text]",
            "entity description": "Specific visual traits, material, and current state.",
            "entity_frame_idx": [when this entity occurs]
        }
    ],
    "relations": [
        {
            "src": "must be from Target Entity IDs",
            "dst": "must be from Target Entity IDs",
            "type": "Relation type",
            "relation description": "explanation with 'because' for causal links",
            "relation_frame_index": [when this relation occurs]
        }
    ],
    "scene": "2-3 sentences summarizing key events and behaviors."
}
[NOTE]
- IDs: Distinguish same-type entities with A, B, C suffixes.
- Prefix: 'entity id' prefix must match its 'type' (e.g., 'location_kitchen').
- Accuracy: Avoid vague words ('some', 'probably'). Use precise scale/state terms.
- Format: Output ONLY the above [JSON Schema].
- Both "entity_frame_idx" and "relation_frame_index" is a list, please populate the list with integer or integers.
- All 'src' and 'dst' must come from "entity id" in entities.
Start JSON:
'''
        
        # print(f"\nframes {frames}\n")
        messages = [{"role": "system", "content": GLOBEL_PEOMPT},
                   {"role": "user", "content": user_content}]
       
        entities, relations, description, scene = self.describe_frame(messages)

        def safe_int_convert(val):
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                match = re.search(r'(\d+)', val)
                if match:
                    return int(match.group(1))
            return None

        if entities:
            for en in entities:
                en['entity_timestamp'] = []
                en['first_seen'] = None
                raw_indices = en.get('entity_frame_idx', [])
                if isinstance(raw_indices, (list, tuple)):
                    for i in raw_indices:
                        frame_val = safe_int_convert(i)
                        if frame_val is not None:
                            en['entity_timestamp'].append(frame_val / float(video_fps))
                    
                    if en['entity_timestamp']:
                        en['entity_timestamp'].sort()
                        en['first_seen'] = en['entity_timestamp'][0]

        if relations:
            for rel in relations:
                rel['relation_timestamp'] = []
                raw_rel_indices = rel.get('relation_frame_index', [])
                if isinstance(raw_rel_indices, (list, tuple)):
                    for index in raw_rel_indices:
                        rel_frame_val = safe_int_convert(index)
                        if rel_frame_val is not None:
                            rel['relation_timestamp'].append(rel_frame_val / float(video_fps))
            
        self.record_frame_index.update(frame_indices)

        # Step 3: 构建子图
        subgraph = {
            'timestamp_range': (
                frame_indices[0] / video_fps if frame_indices else None,
                frame_indices[-1] / video_fps if frame_indices else None
            ),
            'frame_indices': frame_indices,
            'entities': entities,
            'relations': relations,
            'scene_context': scene,
            'raw_descriptions': description
        }
        # print(f"   Vision2GraphParser {subgraph}")
        
        return subgraph

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
1. Identify critical entity (human, object, location, text on the video etc.). Assign unique IDs (person_A, object_cup_B). Describe materials, textures, scale, and immediate physical states (tense, static, occluded).
2. Use Subject-Verb-Object (S-V-O) for micro-actions and interactions. Identify causal traces (e.g., stains implying prior spills).
3. Do NOT report background clutter (e.g., generic trees, walls) unless they provide critical spatial context (e.g., person_A is hiding behind location_wall_B).
4. If ego-view, identify 'ego_player' and describe hand gestures/relative velocity.

[JSON Schema]
{
    "description": "A comprehensive string including: 1.People details; 2.Object states/coordinates; 3.Location/lighting context; 4.Interaction S-V-O chains; 5.Text on the video, including Subtitles, Captions, Overlay text...",
    "entities": [
        {
            "entity id": "unique_id",
            "type": "one of [human, animal, plant, location, object, naturalobject, text]",
            "entity description": "Specific visual traits, material, and current state.",
            "entity_frame_idx": [when this entity occurs] // MUST be a list of PURE INTEGERS.
        }
    ],
    "relations": [
        {
            "src": "must be from Target Entity IDs", // MUST be a STRING, NOT a list.
            "dst": "must be from Target Entity IDs", // MUST be a STRING, NOT a list.
            "type": "relation_type",
            "relation description": "explanation with 'because' for causal links",
            "relation_frame_index": [when this relation occurs] // MUST be a list of PURE INTEGERS.
        }
    ],
    "scene": "2-3 sentences summarizing key events and behaviors."
}

[NOTE]
- "relations" MUST includes "src" and "dst" , the 'src' and 'dst' fields MUST be a single string.
- IDs: Distinguish same-type entities with A, B, C suffixes.
- Prefix: 'entity id' prefix must match its 'type' (e.g., 'location_kitchen').
- Accuracy: Avoid vague words ('some', 'probably'). Use precise scale/state terms.
- Format: Output ONLY the above [JSON Schema].
- Both "entity_frame_idx" and "relation_frame_index" is a list, please populate the list with integer or integers.
Start JSON:
'''
        
        # print(f"\nframes {frames}\n")
        messages = [{"role": "system", "content": GLOBEL_PEOMPT},
                   {"role": "user", "content": user_content}]
       
        entities, relations, description, scene = self.describe_frame(messages)

        if entities:
            for en in entities:
                en['entity_timestamp'] = []
                en['first_seen'] = None
                if en.get('entity_frame_idx', []):
                    for i in en['entity_frame_idx']:
                        en['entity_timestamp'].append(int(i) / float(video_fps))
                    en['entity_timestamp'].sort()
                    en['first_seen'] = en['entity_timestamp'][0]

        if relations:
            for rel in relations:
                rel['relation_timestamp'] = []
                if rel.get('relation_frame_index', []):
                    for index in rel['relation_frame_index']:
                        rel['relation_timestamp'].append(int(index) / float(video_fps))
            
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
























   
        
#         GLOBEL_PEOMPT = f'''
# [System Task]
# As a Forensic Visual Analyst, your goal is to extract visual evidence specifically relevant to the investigation targets. 

# [Analysis Protocol]
# 1. Target-Centric: Identify entities directly mentioned in the {self.question} and {self.candidates} or those in immediate physical contact with them. Assign unique IDs (person_A, object_cup_B). 
# 2. Saliency Filtering: Do NOT report background clutter (e.g., generic trees, walls) unless they provide critical spatial context (e.g., person_A is hiding behind location_wall_B).
# 3. Behavior & Logic: Use Subject-Verb-Object (S-V-O) for micro-actions and interactions. Identify causal traces (e.g., stains implying prior spills).
# 4. Perspective: If ego-view, identify 'ego_player' and describe hand gestures/relative velocity.


# [Output Constraint: STRICT JSON]
# {{
#     "description": "A comprehensive string including: 1.People details; 2.Object states/coordinates; 3.Location/lighting context; 4.Interaction S-V-O chains; 5.Text on the video, including Subtitles, Captions, Overlay text...",
#     "entities": [
#         {{
#             "entity id": "unique_id",
#             "type": "one of [human, animal, plant, location, object, naturalobject]",
#             "entity description": "Specific visual traits, material, and current state."
#         }}
#     ]
# }}

# [Example Requirement]
# - If target is "cup", describe the hand holding it and the table under it. Ignore the window 5 meters away.

# [NOTE]
# - The total number of entities that you extract MUST NO MORE THAN 8 , so you need to select the most relevant and critical ones.
# - If a target entity is NOT visible in this frame, pass. NEVER hallucinate.
# Start JSON:
# '''

#         RELATION_PROMPT = f'''
# [Task]
# Analyze the video entities and descriptions to map a relationship network and summarize the scene.

# [Input Data]
# - Entity IDs: {entity_ids}
# - Detailed Metadata: {entities}
# - Contextual Descriptions: {descriptions}

# [Constraint: Evidence-Based Graphing]
# 1. Relational Sparsity: Only link entities if their relationship directly supports or refutes the Question: {self.question} and Candidates: {self.candidates}. 
# 2. Causal Anchoring: If an action is happening, intuitively observe whether a causal relationship exists. If such a relationship is present, use "because" to link the object state to the behavior.
# 3. Logical Pruning: Discard generic relations like "object_A adjacent_to object_B" unless it's a critical clue.


# [Output Format: STRICT JSON] 
# {{
#     "relations": [
#         {{
#             "src": "must be from Target Entity IDs",
#             "dst": "must be from Target Entity IDs",
#             "type": "action/spatial/causal",
#             "relation description": "critical proof (e.g., 'person_A holds object_knife_B because they are preparing to cook')",
#             "relation_frame_index": [integer_indices]
#         }}
#     ],
#     "scenes": "2-3 sentences summarizing key events and behaviors."
# }}

# [NOTE]
# - The total number of relations that you extract MUST NO MORE THAN 8 , so you need to select the most relevant and critical ones.
# - If a relevant relation is NOT visible in this frame, pass. NEVER hallucinate.
# Start JSON:
# '''
# GLOBEL_PEOMPT = '''
# Perform a panoramic structured analysis of this video frame. You must move beyond the visual center and account for the background, periphery, and all minor entities. Provide an exhaustive description and extract full-scale entities according to the following dimensions:

# 1. Entities & Characters (The 'Who' and 'What'):
# Total Identification: Identify every entities, such as person, creature, vehicle, piece of furniture, electronic device, minute object and so on. Assign a unique identifier to each as entity id (e.g. person_A, object_phone_A).
# Granular Appearance: Describe the material (metal, fabric, liquid), color, texture, estimated scale, and how light interacts with the surface (e.g. The person's coat is blue, the cup is red...).
# Immediate State: Define the physical state of each entity. Is the person 'tense' or 'relaxed'? Is the object 'static', 'motion-blurred', 'powered on', 'damaged', or 'partially occluded'?

# 2. Spatial Layout & Topological Relations (The 'Where'):
# Depth Stratification: Distinguish between the foreground, midground, and background. Note which entities are obscuring others.
# 3D Coordinates: Precisely describe the relative positions (e.g., 'person_A is about 30cm to the left of object_car_B', 'object_cup_C is placed at the geometric center of object_table_A').
# Environmental Context: Define the macro-setting (e.g., 'industrial workshop', 'rainy street') and micro-coordinates (e.g., 'the workstation near the window'). Describe light sources and their impact on shadows.

# 3. Dynamic Behavior & Causal Links (The 'What’s Happening'):
# Atomic Actions: Describe ongoing micro-movements (e.g., 'fingers slightly curled', 'center of gravity shifted to the left') rather than vague concepts.
# Interaction Logic (S-V-O): Use a 'Subject-Verb-Object' structure for all interactions (e.g., 'A person who wears blue  is applying force to pull Handle').
# Causal Traces: Identify clues that imply prior actions (e.g., 'water stains on the floor imply a recent spill').

# 4. First-Person Perspective (Ego-View - If Applicable):
# Self-Awareness: If the view is first-person, refer to the camera wearer as 'ego_player'. Describe visible hand gestures, the grip force on held objects, and the sense of relative velocity against the environment.

# Ensure the output strictly follows the JSON format below:
# {
#     "description": "comprehensive description",
#     "entities": [
#         {"entity id": "unique_id (e.g. person_A, object_phone_A, location_bedroom_B)", "type":"the type of this entity (e.g. human, location)", "entity description": "detailed visual traits and current state about this entity"}
#     ]
# }
# Note: 
# It's important that the output MUST be the above JSON format.
# description should include: People (If this frame has)--Appearance details, micro-action specifics. Objects--Material & state, spatial coordinates, physical contact points with other entities. Location & Context--Setting type, lighting conditions, atmospheric description, and dynamic trends. Interaction Chains--All S-V-O mappings, causal chains, and potential movements.
# A,B... are used to distinguish different in the same type, such as different people.
# the 'type' of entity and prefix of unique_id must come from ["human", "animal", "plant", "location", "object","naturalobject"] (e.g. the prefix of 'location_bedroom' is 'location' )
# entities must include three part: entity id, type and entity description.
# Strictly avoid vague terms like 'some', 'probably', or 'things'. Act as a forensic investigator recording a scene: objective, precise, and exhaustive in capturing the data behind every pixel.'''

GLOBEL_PEOMPT = '''
[System Task]
As a forensic visual analyst, perform a panoramic structured analysis of the provided video frame. Your goal is to capture every pixel's data with objective precision.

[Analysis Dimensions]
1. Entities & Characters: Identify every entity (human, object, location, etc.). Assign unique IDs (person_A, object_cup_B). Describe materials, textures, scale, and immediate physical states (tense, static, occluded).
2. Spatial & Topology: Stratify depth (foreground/midground/background). Define 3D relative positions and environmental micro-coordinates.
3. Behavior & Logic: Use Subject-Verb-Object (S-V-O) for micro-actions and interactions. Identify causal traces (e.g., stains implying prior spills).
4. Perspective: If ego-view, identify 'ego_player' and describe hand gestures/relative velocity.

[Output Constraint: MUST BE STRICT JSON]
Your entire response must be a single, valid JSON object. Do not include markdown code blocks (like ```json), conversational filler, or analysis outside the JSON structure.

[JSON Schema]
{
    "description": "A comprehensive string including: 1.People details; 2.Object states/coordinates; 3.Location/lighting context; 4.Interaction S-V-O chains; 5.Text on the video, including Subtitles, Captions, Overlay text...",
    "entities": [
        {
            "entity id": "unique_id (e.g., person_A, object_phone_B)",
            "type": "one of [human, animal, plant, location, object, naturalobject]",
            "entity description": "Specific visual traits, material, and current state."
        }
    ]
}

[Strict Requirements]
- IDs: Distinguish same-type entities with A, B, C suffixes.
- Prefix: 'entity id' prefix must match its 'type' (e.g., 'location_kitchen').
- Accuracy: Avoid vague words ('some', 'probably'). Use precise scale/state terms.
- Completeness: No entity or minor interaction in the periphery should be missed.

[Example Response Format]
{"description": "In a dimly lit kitchen, person_A (wearing a red wool sweater) is standing 50cm from location_table_A. Their right hand is applying downward pressure on object_knife_A...", "entities": [{"entity id": "person_A", "type": "human", "entity description": "Adult male, red sweater, relaxed posture, focal point."}, {"entity id": "location_kitchen_A", "type": "location", "entity description": "Indoor setting, tile floor, fluorescent overhead light."}]}

Start your JSON output now:
'''

#  RELATION_PROMPT = f'''
# [Task]
# Analyze the video entities and descriptions to map a relationship network and summarize the scene.

# [Input Data]
# - Target Entity IDs: {entity_ids}
# - Detailed Metadata: {entities}
# - Contextual Descriptions: {descriptions}

# [Relation Type]
# 1. left_of, inside, on_top_of, adjacent_to, etc.
# 2. holding, pulling, moving_towards, etc.
# 3. caused_by, activated_by, belongs_to -> MUST include "because" in description.
# 4. watching, communicating_with, etc.

# [Output Format: STRICT JSON] 
# {{
#     "relations": [
#         {{
#             "src": "must be from Target Entity IDs",
#             "dst": "must be from Target Entity IDs",
#             "type": "relation_type",
#             "relation description": "explanation with 'because' for causal links",
#             "relation_frame_index": [integer_indices],
#         }}
#     ], 
#     "scenes": "2-3 sentences summarizing key events and behaviors."
# }}

# [Constraints]
# - All 'src' and 'dst' must match the provided Entity IDs exactly.
# - Output ONLY the JSON object. No markdown, no filler.

# Start JSON:
# '''

# - 'relation_frame_index' must be integers derived from the input metadata, comes from 'entity_frame_idx' in 'Detailed Metadata' and 'frame_idx' in 'Contextual Descriptions'.
# - Do not just focus on the 'ego_player' or central characters. Please map all relationships, even those between background objects.

#         RELATION_PROMPT = f'''
# Analyze the provided video entities and their descriptions to map a comprehensive network of relationships and summarize a simple scenes in 2-3 sentences. You must look beyond simple interactions and capture the full environmental and logical topology.
# Target Entities: {entities}, the only source of src and dst. 
# Contextual Descriptions: {descriptions} 

# Extraction Dimensions:
# 1.Spatial Relationships (Static/Positional):
# Identify the 3D spatial layout. Use types like left_of, right_of, inside, on_top_of, adjacent_to, or occluding. (e.g. person_A is inside object_car_B).

# 2.Actionable & Kinetic Relationships (Dynamic):
# Describe physical forces and movements. Use types like holding, pulling, colliding_with, moving_towards, or following. Specify the intensity or mode of action where possible.

# 3.Causal & Functional Relationships (Logical):
# Identify if one entity's state change is triggered by another. Use types like caused_by, activated_by, or belongs_to (part-whole relationship). (e.g. The 'Open' state of object_door_B is caused_by person_A).

# 4.Social & Intentional Relationships (High-level):
# If applicable, identify group dynamics or focus of attention, such as watching, communicating_with, or competing_with.

# Ensure the output strictly follows the JSON format below:
# {{
#     "relations":[
#        {{"src": "entity id",
#         "dst": "entity id",
#         "type": "specific_relation_type", 
#         "category": "spatial | kinetic | causal | social",
#         "relation description": "Short natural language explanation of the link",
#         "relation_frame_index": [], 
#         "confidence": 0.0-1.0}}
#         ],
#     "scenes":"summarized scenes that include Key events, Important interactions, Suspicious or unusual behaviors"

# }}
# Note: 
# The scenes must focus on Key events Important interactions Suspicious or unusual behaviors.
# 'Target Entities' and 'Contextual Descriptions' that I provide to you are lists that include dictionaries in the same format, all of which contain the video frame index. It is the key of 'entity_frame_idx' in 'Target Entities' and the key of 'frame_idx' in 'Contextual Descriptions'.The relation_frame_index you generate must be derived from the values corresponding to the keys of 'entity_frame_idx' and 'frame_idx'. Of course, you can generate multiple indices that meet the requirements.
# 'relation_frame_index' record from which video frames you extracted this relation, please fill the list with the integer form of the correct frame indices.
# Do not include any conversational filler and ensure all src and dst IDs match the provided entity_ids exactly.
# If relations has causal relationship, 'relation description' must include str 'because'.
# Do not just focus on the 'ego_player' or central characters. Map relationships between background objects if they define the scene (e.g., 'Lamp' on_top_of 'Table').
# Provide the exact frame index for each relation.
# For every relation, consider if the inverse is also significant (e.g., 'person_A is near object_car_B' implies 'object_car_B is near person_A').
# '''


# GRAPH_PROMPT = 
# Please analyze the given video and extract key information in a structured JSON format in English. Identify and describe:

# Entities & Descriptions: List all distinct objects, people, or animals. For each, provide a "description" that captures their visual traits (e.g., color, size) and their current state (e.g., "standing still", "opened", "broken").
# Actions & Relations: Describe interactions between entities or an entity's movement. Focus on "Subject - Action - Object" relationships to define how entities relate to one another.
# Scenes & Context: Identify and describe the locations, environments, or contexts where the events occur. (e.g., lighting, atmosphere, or layout).
# If the video is filmed from a first-person point of view, please describe the camera wearer as "me" and detail actions or interactions from this perspective.
# Ensure the output strictly follows the JSON format below:
# {
#     "entities": [
#         {"entity name": "unique_id", "description": "detailed visual traits and current state"}
#     ],
#     "actions": [
#         {"entity name": "subject_id", "action description": "subject does [action] to [object_id if exists]"}
#     ],
#     "scenes": [
#         {"location": "setting name", "context": "environmental details and scene state"}
#     ]
# }
# The "entity name" used in "actions" must match exactly with those defined in "entities".
# In "actions", clearly specify "relations" (e.g., "A is holding B", "A is standing next to B").
# Each section should be detailed but concise, capturing all relevant interactions and contextual elements from the video. 
# Avoid unnecessary text outside the JSON output.

# def parse(self, frames, frame_indices):
#     """
#     输入: 一组帧 + 索引
#     输出: 结构化子图
#     """
#     # VLM生成详细描述
#     descriptions = []
#     for i, frame in enumerate(frames):
#         generated_text = self.describe_frame(    
#             frame,
#             prompt="""
# Describe this video frame in detail:
# - Who are present? (people, their appearance, actions)
# - What objects are visible? (important items, their state)
# - Where is this? (location, setting)
# - What's happening? (events, interactions)

# Use structured format:
# People: ...
# Objects: ...
# Location: ...
# Events: ...
# """
#         )
#         descriptions.append({
#             'frame_idx': frame_indices[i],
#             'timestamp': frame_indices[i] / 30.0,  # 假设30fps
#             'description': generated_text
#         })
    
#     # 提取结构化信息
#     entities = self.extract_entities(descriptions)
#     relations = self.extract_relations(descriptions, entities)
#     scene_summary = self.summarize_scene(descriptions)
    
#     # Step 3: 构建子图
#     subgraph = {
#         'timestamp_range': (
#             frame_indices[0] / 30.0,
#             frame_indices[-1] / 30.0
#         ),
#         'frame_indices': frame_indices,
#         'entities': entities,
#         'relations': relations,
#         'scene_context': scene_summary,
#         'raw_descriptions': descriptions
#     }
    
#     return subgraph

# def describe_frame(self, frame, prompt):
#     """
#     使用Qwen2.5VL描述单帧
#     """
#     # 确保frame是PIL图像或numpy数组
#     if isinstance(frame, torch.Tensor):
#         frame = frame.permute(1, 2, 0).numpy()  # [H,W,C]
#         frame = (frame * 255).astype('uint8')
    
#     # 构建消息
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": frame
#                 },
#                 {
#                     "type": "text", 
#                     "text": prompt
#                 }
#             ]
#         }
#     ]
    
#     # 预处理
#     text = self.processor.apply_chat_template(
#         messages, 
#         tokenize=False, 
#         add_generation_prompt=True
#     )
#     inputs = self.processor(
#         text=[text],
#         images=[frame],
#         padding=True,
#         return_tensors="pt"
#     )
    
#     # 生成描述
#     with torch.no_grad():
#         generated_ids = self.vlm.generate(
#             **inputs,
#             max_new_tokens=512,
#             do_sample=False
#         )
        
#     generated_text = self.processor.batch_decode(
#         generated_ids, 
#         skip_special_tokens=True
#     )[0]
    
#     return generated_text

#     def extract_entities(self, descriptions):
#         """
#         从描述中提取实体
#         使用LLM进行结构化抽取
#         """
#         combined_desc = "\n\n".join([d['description'] for d in descriptions])
        
#         prompt = f"""
# From the following video frame descriptions, extract all entities:

# {combined_desc}

# For each entity, provide:
# - id: unique identifier (e.g., person_A, object_knife)
# - type: person / object / location
# - description: brief description
# - first_seen: timestamp when first appeared

# Output in JSON list format.
# """
        
#         response = self.vlm.generate(prompt=prompt, max_tokens=1024)
#         entities = self.parse_json_list(response)
        
#         return entities

#     def extract_relations(self, descriptions, entities):
#         """
#         抽取实体间的关系
#         """
#         entity_ids = [e['id'] for e in entities]
        
#         prompt = f"""
# Given these entities: {entity_ids}

# And these descriptions:
# {descriptions}

# Extract relationships between entities:
# - src: source entity id
# - dst: destination entity id  
# - type: action/spatial/causal (e.g., "holding", "near", "caused")
# - timestamp: when this relation occurs
# - confidence: 0.0-1.0

# Output as JSON list.
# """
        
#         response = self.vlm.generate(prompt=prompt, max_tokens=1024)
#         relations = self.parse_json_list(response)
        
#         return relations 

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

# system_prompt = f"""You are a Video Forensic Investigator. Your task is to extract irrefutable visual evidence to solve complex video reasoning problems.

# [INVESTIGATION PROTOCOL - MANDATORY]
# 1. **KEYWORD TRIGGER**: If the question or options mention specific names (e.g., 'Benjamin'), objects, or actions, and they are NOT in the current summary, you MUST use `choose frames between START_FRAME and END_FRAME` to find them. Do NOT assume they don't exist.
# 2. **TIMESTAMP ANCHORING**: If a timestamp is mentioned (e.g., '0:28'), you MUST use `get frame number at time 0:28` or `choose frames between START_FRAME and END_FRAME` near that point immediately. You ARE FORBIDDEN from answering before inspecting the specific time mentioned.
# 3. **NO EVIDENCE, NO ANSWER**: If the current reasoning leads to "Information not available," your ONLY allowed action is to SEARCH. "None of the above" or guessing is a strict violation of your protocol.
# 4. **EXPLANATION ALIGNMENT**: Base your reasoning on subtle cues. For example: intimate terms like 'bro' or sending medicine imply a 'friend' relationship; low battery prompts near a promised call imply 'charging for the call'.

# [ACTION CATEGORIES]
# 1. `choose frames between START_FRAME and END_FRAME`: Primary tool for visual search. START_FRAME, END_FRAME MUST BE integer OR MM:SS.
# 2. `get frame number at time MM:SS`: Tool for time-to-frame conversion. 
# - CONSTRAINT: 00:00 <= MM:SS <= {max_time_str}.
# 3. `output answer: OPTION`: Final step ONLY. The OPTION MUST be a single letter from {candidates}.

# [TECHNICAL METADATA]
# - Video FPS: {video_fps} | Total Frames: {total_frames} | Duration: {max_time_str}.
# - Formula: Frame = seconds * {video_fps}. (e.g., 0:28 = 28 * {video_fps} = {int(28 * video_fps)}).
# - Constraints: 0 <= START_FRAME < END_FRAME <= {total_frames-1} AND (END_FRAME - START_FRAME) >= {num_frames_to_sample}.

# [RESPONSE FORMAT]
# <think> 1. Identify missing info (e.g., Where is Benjamin?). 2. Calculate target frames. 3. Plan search. </think> <action>choose frames between X and Y</action> OR <action>get frame number at time MM:SS</action> OR <action>output answer: OPTION</action>
# """

# system_prompt = f"""
# You are an expert AI assistant that answers questions about a video by iteratively analyzing the graph information generated by it.
# Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.

# [INVESTIGATION PROTOCOL - MANDATORY]
# 1. **KEYWORD TRIGGER**: If the QUESTION or CANDIDATE OPTIONS mention specific names, objects, or actions, and they are NOT in the current summary, you MUST use `choose frames between START_FRAME and END_FRAME` to find them. 
# 2. **TIMESTAMP ANCHORING**: If a timestamp is mentioned (e.g., '0:28'), you MUST use `get frame number at time 0:28` or `choose frames between START_FRAME and END_FRAME` near that point immediately. You ARE FORBIDDEN from answering before inspecting the specific time mentioned.
# 3. **NO EVIDENCE, NO ANSWER**: If the current reasoning leads to "Information not available," your ONLY allowed action is to `get frame number at time MM:SS` or `choose frames between START_FRAME and END_FRAME`. "None of the above" or guessing is a strict violation of your protocol.
# 4. **EXPLANATION ALIGNMENT**: Base your reasoning on subtle cues. For example: intimate terms like 'bro' or sending medicine imply a 'friend' relationship; low battery prompts near a promised call imply 'charging for the call'.
# 5. **CANDIDATE-DRIVEN SEARCH**: Treat each option as a hypothesis. Search for visual cues that could support or refute specific options (e.g., if options include 'doctor', search for medical equipment; if 'neighbor', search for outdoor/doorway interactions). Use the keywords in {candidates} to guide your search.

# [ACTION CATEGORIES]
# 1. `choose frames between START_FRAME and END_FRAME`: Primary tool for visual search. START_FRAME, END_FRAME MUST BE integer OR MM:SS, no float or other.
# 2. `get frame number at time MM:SS`: Tool for time-to-frame conversion. 
# - CONSTRAINT: 00:00 <= MM:SS <= {max_time_str}.
# 3. `output answer: OPTION`: Final step ONLY. The OPTION MUST be a single letter from {list(candidates.keys())}.

# [TECHNICAL METADATA]
# - Video FPS: {video_fps} | Total Frames: {total_frames} | Duration: {max_time_str}.
# - Formula: Frame = seconds * {video_fps}. (e.g., 0:28 = 28 * {video_fps} = {int(28 * video_fps)}).
# - Constraints: 0 <= START_FRAME < END_FRAME <= {total_frames-1} AND (END_FRAME - START_FRAME) >= {num_frames_to_sample}.

# [RESPONSE FORMAT]
# <think> 1. Analyze keywords in both question and ALL options. 2. Formulate hypotheses (e.g., If F is true, I should see...). 3. Calculate frames and plan search. </think> <action>choose frames between START_FRAME and END_FRAME</action> OR <action>get frame number at time MM:SS</action> OR <action>output answer: OPTION</action>
# """

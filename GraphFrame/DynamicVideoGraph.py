import networkx as nx
from collections import defaultdict
import torch, re
import json
import numpy as np
from Vgent.utils.prompts import *

  
class DynamicVideoGraph:
    """
    动态增长的视频知识图谱
    """
    def __init__(self, graph, embedding_tokenizer, embedding_model):
        # 核心图结构
        self.graph = graph # 有向多重图
        self.observations = []  # 所有子图的列表
        self.entity_index = {}  # entity_id -> node_data
        self.temporal_index = defaultdict(list)  # timestamp -> entities
        self.relation_index = defaultdict(list)  # relation_type -> edges's information e.g. who -> who, time
        
        self.entity_embeddings = {}  # 用于视觉相似度对齐

        self.embedding_tokenizer = embedding_tokenizer
        self.embedding_model = embedding_model
        
    def add_subgraph(self, subgraph, turn_id):
        """
        融合新的子图到全局图
        """
        print(f"  📊 Adding subgraph from Turn {turn_id}...")
        
        # 1. 添加实体（带对齐）
        entity_mapping = {}  # new_id -> aligned_id
        if subgraph['entities']:
            for entity in subgraph['entities']:
                aligned_id = self.align_or_add_entity(entity, turn_id)
                entity_mapping[entity['entity id']] = aligned_id
        
        # 2. 添加关系

        if subgraph['relations']:
            print(f"\nsubgraph['relations'] {subgraph['relations']}")
            for relation in subgraph['relations']:
                raw_src = relation.get('src', None)
                raw_dst = relation.get('dst', None)

                if not raw_src:
                    continue 

                is_unary = False
                if raw_dst is None:
                    raw_dst = 'scene_context'
                    is_unary = True

                for raw_id in [raw_src, raw_dst]:
                    if raw_id not in entity_mapping:
                        if raw_id in self.graph:
                            entity_mapping[raw_id] = raw_id
                        else:
                            self.graph.add_node(
                                raw_id,
                                type="scene_context" if raw_id == 'scene_context' else "none",
                                entity_description="context or background object",
                                first_seen=turn_id,
                                last_seen=turn_id,
                                entity_occur_frame_index=[None],
                                attributes={}
                            )
                            entity_mapping[raw_id] = raw_id

                src = entity_mapping[raw_src]
                dst = entity_mapping[raw_dst]

                self.graph.add_edge(
                    src, dst,
                    key=f"{turn_id}_{relation.get('type', 'none')}_{relation.get('relation_timestamp', [])}",
                    type=relation.get('type', 'none'),
                    relation_timestamp=relation.get('relation_timestamp', []),
                    relation_description=relation.get('relation description', 'none'),
                    relation_occur_frame_index=relation.get('relation_frame_index', []),
                    turn_id=turn_id,
                    is_unary=is_unary # 额外打个标签，方便以后查询
                )
            
                # 更新关系索引
                rel_type = relation.get('type', 'none')
                self.relation_index[rel_type].append({
                    'src': src,
                    'dst': dst,
                    'timestamp': relation.get('relation_timestamp', []),
                    'turn': turn_id,
                    'is_unary': is_unary
                })
        
        # 更新时序索引
        start, end = subgraph['timestamp_range']
        if start and end:
            self.temporal_index[int(start)].extend(list(entity_mapping.values()))  # 是values()而非keys()
        
        # 4. 保存子图
        self.observations.append({
            'turn': turn_id,
            'subgraph': subgraph,
            'entity_mapping': entity_mapping
        })
        
        print(f"\nGraph now has {self.graph.number_of_nodes()} entities, "
              f"{self.graph.number_of_edges()} relations\n")
    
    def align_or_add_entity(self, new_entity, turn_id):
        """
        实体对齐：判断是否为已存在实体
        """
        # 1. 精确匹配（如果ID格式一致）
        if new_entity['entity id'] in self.entity_index:
            if isinstance(self.graph.nodes[new_entity['entity id']]['first_seen'], float) and isinstance(new_entity['first_seen'], float) and self.graph.nodes[new_entity['entity id']]['first_seen'] >= new_entity['first_seen']:
                self.graph.nodes[new_entity['entity id']]['first_seen'] = new_entity['first_seen']
            self.graph.nodes[new_entity['entity id']]['last_seen'] = turn_id
            self.graph.nodes[new_entity['entity id']]['entity_description'] += new_entity['entity description']
            self.graph.nodes[new_entity['entity id']]['entity_occur_frame_index'].append(new_entity['entity_frame_idx'])
            return new_entity['entity id']
        
        # 2. 模糊匹配（同类型实体）
        same_type_entities = [
            (eid, data) for eid, data in self.graph.nodes(data=True)
            if data.get('type', 'none') == new_entity['type']
        ]
        
        for eid, data in same_type_entities:
            # 计算相似度
            entity_description_sim = self.compute_entity_similarity([new_entity['entity description']], [data['entity_description']], self.embedding_model, self.embedding_tokenizer, return_all=True)
            similarity_idx = max(range(len(entity_description_sim)), key=lambda i: entity_description_sim[i])
            similarity = entity_description_sim[similarity_idx]
            
            if similarity > 0.8:  
                print(f"\nAligned {new_entity['entity id']} to existing {eid}")
                if isinstance(data['first_seen'], float) and isinstance(new_entity['first_seen'], float) and data['first_seen'] >= new_entity['first_seen']:
                    data['first_seen'] = new_entity['first_seen']
                data['entity_description'] = data['entity_description'] + new_entity['entity description']  #这里原来调用的是self.merge_descriptions,我暂时直接相加，因为实现逻辑有点简单
                data['last_seen'] = turn_id
                data['entity_occur_frame_index'].append(new_entity['entity_frame_idx'])
                return eid
        
        # 3. 新实体
        entity_id = new_entity['entity id']
        self.graph.add_node(
            entity_id,
            type=new_entity.get('type', 'none'),
            entity_description=new_entity.get('entity description', 'none'),
            first_seen=new_entity.get('first_seen', turn_id),
            last_seen=turn_id,
            entity_occur_frame_index = new_entity.get('entity_frame_idx', []),
            attributes=new_entity.get('attributes', {})
        )
        self.entity_index[entity_id] = new_entity # new_entity是一个字典
        
        # print(f"\nAdded new entity: {entity_id}")
        return entity_id
    
    def compute_entity_similarity(self, query_list, key_list, embedding_model, tokenizer, return_all=False):
        """
        通过描述计算两个实体描述的相似度
        """
        encoded_input = tokenizer(query_list + key_list, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = embedding_model(**encoded_input)
            embeddings = model_output[0][:, 0]
        query_emb = torch.nn.functional.normalize(embeddings[:len(query_list)], p=2, dim=1)
        key_emb = torch.nn.functional.normalize(embeddings[len(query_list):], p=2, dim=1)
        sims = query_emb @ key_emb.T
        if return_all:
            return sims[0]
        else:
            return torch.mean(sims)
    
    def merge_descriptions(self, old_desc, new_desc):
        """
        合并两个描述，保留更详细的信息
        """
        # 简单策略：取更长的 
        return new_desc if len(new_desc) > len(old_desc) else old_desc
    
    def summarize(self):
        """
        生成图谱的文本摘要
        """
        summary = {
            'num_entities': self.graph.number_of_nodes(),
            'num_relations': self.graph.number_of_edges(),
            'entities': [],
            'relations': [],
            'temporal_chain': []
        }

        # 实体列表
        for node, data in self.graph.nodes(data=True):
            summary['entities'].append(f"{node} ({data.get('type', 'none')}): {data.get('entity_description', 'none')} entity_occur_frame_index -> {data.get('entity_occur_frame_index', [])}")
        
        # 关系列表（去重）
        seen_relations = set()
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            rel_str = f"{u} --[{data.get('type', 'none')}, t={data.get('relation_timestamp', 'none')}s]--> {v} (relation description: {data.get('relation_description', 'none')}, relation_occur_frame_index: {data.get('relation_occur_frame_index', [])})"
            if rel_str not in seen_relations:
                summary['relations'].append(rel_str)
                seen_relations.add(rel_str)
        
        # 时序链
        sorted_times = sorted(self.temporal_index.keys())
        if sorted_times:
            for t in sorted_times:
                entities_at_t = self.temporal_index.get(t, [])
                summary['temporal_chain'].append(f"t={t}s: {entities_at_t}")
        
        return summary
    
    def extract_questin_candidates_keywords(self, question, candidates, video_inputs=None):
        reason_prompt = REASONING_PROMPT.format(query=question, candidates=candidates)
        flag = True
        count = 0 
        llm_info = None
        while flag and count < 5: 
            try:
                response = self.mllm_response(self.video_llm, self.processor, self.image_processor, reason_prompt, None, None, max_new_tokens=256)
                llm_info = json.loads(response.replace("```json", "").replace("```","").strip())
                flag = False
            except:
                count += 1
                continue
        
        query_list = llm_info["keywords"] if llm_info is not None and "keywords" in llm_info else []
        query_list = query_list + list(candidates.values())   # 我的数据集是字典，所以需要将字典的值转成列表
        query_list = list(set(query_list))

        return query_list, llm_info
    
    def extract_choices(self, question, candidates):
        candidates = list(candidates.values())
        if "(1)" in question or "(a)" in question:
            pattern = r'\(([a-zA-Z0-9]+)\)\s*(.+?)(?=\s*\([a-zA-Z0-9]+\)|$)'
            matches = re.findall(pattern, question, flags=re.DOTALL)
            query_list = [c[1].strip() for c in matches]
        elif re.search(r"\d+\.", question):
            pattern = r"\d+\.\s+([^\n]+)"
            matches = re.findall(pattern, question)
            query_list = [match.strip() for match in matches]
        elif "-->" in candidates[0]:
            choices = candidates[0].split("-->")
            query_list = [choice.strip() for choice in choices]
        elif len(candidates[0].split(",")) > 2:
            query_list = []
            for candidate in candidates:
                choices = re.sub(r'^[A-Za-z0-9]+\.\s*', '', candidate)
                choices = choices.rstrip('.')
                query_list.extend([item.strip().lower() for item in choices.split(',') if item.strip()])
            query_list = list(set(query_list))
        elif len(candidates[0].split(",")) > 1 and 'and' in candidates[0]:
            query_list = []
            for candidate in candidates:
                choices = re.sub(r'^[A-Za-z0-9]+\.\s*', '', candidate)
                choices = choices.rstrip('.')
                choices = choices.replace(' and ', ',')
                query_list.extend([item.strip().lower() for item in choices.split(',') if item.strip()])
            query_list = list(set(query_list))
        elif re.search(r"[①-⑩]", question):  # 新增
            pattern = r'[①-⑩]\s*(.+?)(?=\s*[①-⑩]|$)'
            matches = re.findall(pattern, question, flags=re.DOTALL)
            query_list = [match.strip() for match in matches]
        else:
            query_list = candidates
        return query_list

    def compute_text_similarity(self, query_list, key_list, embedding_model, tokenizer, return_all=False):
        encoded_input = tokenizer(query_list + key_list, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = embedding_model(**encoded_input)
            embeddings = model_output[0][:, 0]
        query_emb = torch.nn.functional.normalize(embeddings[:len(query_list)], p=2, dim=1)
        key_emb = torch.nn.functional.normalize(embeddings[len(query_list):], p=2, dim=1)
        sims = query_emb @ key_emb.T
        if return_all:
            return sims
        else:
            return torch.mean(sims)

    def allocate_node(self, video_graph, query_list, embedding_model, tokenizer, threshold=0.75, top_k=3):
        node_scores = {}
        node_list = []

        def update_node_score(node_id, score):
            score_val = float(score)
            if score_val > threshold:
                if node_id not in node_scores or score_val > node_scores[node_id]:
                    node_scores[node_id] = score_val

        for key in video_graph:  # 和实体的相似度
            score = self.compute_text_similarity(query_list, [key], embedding_model, tokenizer)
            update_node_score(key, score)

        for (node, data) in video_graph.nodes(data=True): # 和实体描述的相似的
            if data.get('subtitles') is None:
                key_list = f"{node} ({data.get('type', 'none')}): {data.get('entity_description', 'none')}"
            else:
                key_list = f"{node} ({data.get('type', 'none')}): {data.get('entity_description', 'none')}" + data.get('subtitles', [])
            score = self.compute_text_similarity(query_list, [key_list], embedding_model, tokenizer)
            update_node_score(node, score)
        
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            rel_str = f"{u} --[{data.get('type', 'none')}, t={data.get('relation_timestamp', 'none')}s]--> {v} (relation description: {data.get('relation_description', 'none')})"
            if data.get('subtitles') is None:
                key_list = rel_str
            else:
                key_list = rel_str + data.get('subtitles', [])
            score = self.compute_text_similarity(query_list, [key_list], embedding_model, tokenizer)
            update_node_score(u, score)
            update_node_score(v, score)    

        sorted_nodes = sorted(node_scores.items(), key=lambda item: item[1], reverse=True)
        final_node_list = [node for node, score in sorted_nodes]
        if len(final_node_list) > top_k:
            final_node_list = final_node_list[:top_k]

        print(f"Final filtered node_list (Top-{top_k}): {final_node_list}")
        return final_node_list
    
    def retrieve_nodes(self, question, query_list, video_inputs, candidates, subtitles, llm_info):
        indices = None
        if "subtitle" in question.lower() and subtitles is not None and re.findall(r"'((?:[^']|(?<=\w)'(?=\w))*)'", question):
            query_subtitle = re.findall(r"'((?:[^']|(?<=\w)'(?=\w))*)'", question)
            indices = []
            for time, text in subtitles:
                if text in query_subtitle:  
                    indices.append(time)
            node_list = []
        # elif 'beginning' in question.lower() or 'at the start of' in question.lower():
        #     node_list = [i for i in range(3)]
        # elif 'at the end of the video' in question.lower():
        #     node_list = [i for i in range(max(round(np.ceil(len(video_inputs[0]) / self.args.chunk_size)) - 3, 0), round(np.ceil(len(video_inputs[0]) / self.args.chunk_size)))]
        # elif video_graph is None:
        #     node_list = list(range(round(np.ceil(len(video_inputs[0]) / self.args.chunk_size)))) if (llm_info is not None and "tool" in llm_info and llm_info["tool"] in ["action counting", "order"] and self.args.task == 'mlvu') or len(video_inputs[0]) <= 128 else []
        else:
            if "order" in question.lower():
                query_list = self.extract_choices(question, candidates)
            query_list.append(question)
            node_list = self.allocate_node(self.graph, query_list, self.embedding_model, self.embedding_tokenizer)
        #     key_list = []
        #     for node_id in node_list:
        #         node_data = video_graph.nodes[node_id]
        #         if node_data.get('subtitles') is None:
        #             key_list.append("; ".join(node_data.get('entities', '')) + "; ".join(node_data.get('actions', '')) + "; ".join(node_data.get('scenes', '')))
        #         else:
        #             key_list.append("; ".join(node_data.get('entities', '')) + "; ".join(node_data.get('actions', '')) + "; ".join(node_data.get('scenes', '')) + "; ".join(node_data.get('subtitles', '')))
        #     sims = compute_text_similarity(query_list, key_list, self.embedding_model, self.embedding_tokenizer, return_all=True)
        #     sorted_indices = torch.argsort(torch.mean(sims, dim=0), descending=True)
        #     node_list = [node_list[i] for i in sorted_indices]
        # return {"nodes": node_list[:self.args.n_retrieval], "indices": indices}
        return {"nodes": node_list, "indices": indices}

    def query_evidence_chains(self, question, candidates, subtitles, video_inputs=None):
        """
        根据问题查询相关的证据链
        """
        # 简单实现：提取问题中的关键实体，找到相关路径
        # 例如问题 "Who took the knife?"
        # 提取: "knife" -> 找到object_knife
        # 查询: 谁与object_knife有"take"关系
        
        evidence_chains = []
        
        # 1. 根据问题筛选出实体，即节点
        query_list, llm_info = self.extract_questin_candidates_keywords(question, candidates, video_inputs)
        question_relevent_entities = self.retrieve_nodes(question, query_list, video_inputs, candidates, subtitles, llm_info)
        start_nodes = question_relevent_entities.get("nodes", [])

        if not start_nodes:
            return [{"type": "system", "chain": "No direct entity matches found in graph."}]
        
        # 2. 对每个实体，找相关路径
        for entity_id in start_nodes:
            if entity_id not in self.graph:
                continue
            
            # 入边：谁对这个实体做了什么
            in_edges = self.graph.in_edges(entity_id, data=True) # 没有key=True将返回三元组
            if in_edges:
                for src, dst, data in in_edges:
                    evidence_chains.append({
                        'type': 'direct',
                        'chain': f"{src} {data.get('type', 'none')} {dst} at {data.get('relation_timestamp', 'none')}s, simple description: {data.get('relation_description', 'none')}"
                    })
        
            # 出边：这个实体对谁做了什么
            out_edges = self.graph.out_edges(entity_id, data=True)
            if out_edges:
                for src, dst, data in out_edges:
                    evidence_chains.append({
                        'type': 'direct',
                        'chain': f"{src} {data.get('type', 'none')} {dst} at {data.get('relation_timestamp', 'none')}s, simple description: {data.get('relation_description', 'none')}"
                    })
        
        # 3. 多跳推理（可选）
        if len(start_nodes) >= 2:
            for i in range(len(start_nodes)):
                for j in range(i + 1, len(start_nodes)):
                    try:
                        path = nx.shortest_path(self.graph, start_nodes[i], start_nodes[j])
                        if 1 < len(path) <= 4: # 限制推理深度，防止过度联想
                            path_str = f"[{path[0]}]"
                            for i in range(len(path) - 1):
                                u, v = path[i], path[i+1]
                                edge_data = self.graph.get_edge_data(u, v) 

                                if edge_data:
                                    actions = []
                                    for key, dict in edge_data.items():
                                        action = dict.get('type', 'interact')
                                        actions.append(action)
                                    
                                    path_str += f" --({actions})--> [{v}]"
                                else:
                                    path_str += f" --(unknown)--> [{v}]"
                            evidence_chains.append({
                                'type': 'reasoning_path',
                                'chain': f"Causal/Logic Path found: {path_str}"
                            })
                    except nx.NetworkXNoPath:
                        continue
         
        return evidence_chains
    
    def find_suspicious_patterns(self, threshold_seconds):
        """
        悬疑/异常检测逻辑：
        1. 行为突变：实物前后动作属性不匹配。
        2. 空间矛盾：同一时间出现在两个地方。
        3. 隐秘关联：通过不显眼的中间物（如同一个杯子）连接的两个人。
        """
        patterns = []
        
        # 模式1：检测行为突变 (Anomaly Detection)
        for node, node_data in self.graph.nodes(data=True):
            actions = []
            for u, v, data in self.graph.edges(node, data=True):
                actions.append((data.get('relation_timestamp', []), data.get('type', 'none')))
            
            actions.sort() 
            
            # 这里简化为检测动作类型的快速切换
            if len(actions) >= 3:
                patterns.append({
                    'type': 'behavior_frequency_alert',
                    'entity': node,
                    'description': f"High frequency of actions for {node} detected: {actions}"
                })

        # 模式2：隐藏连接，寻找 A -> B <- C 结构，其中 B 通常是 Object 或 Scene
        all_nodes = list(self.graph.nodes())
        for i in range(len(all_nodes)):
            for j in range(i + 1, len(all_nodes)):
                node_a, node_c = all_nodes[i], all_nodes[j]
    
                # 寻找共同交互过的节点
                common_objs = set(self.graph.neighbors(node_a)) & set(self.graph.neighbors(node_c))
                # 过滤掉 location 等公共节点（像商场，客厅），专注具体的 object
                suspicious_objs = [obj for obj in common_objs if self.graph.nodes[obj].get('type', 'none') != 'location' and self.graph.nodes[obj].get('type', 'none') != 'none']
                
                if suspicious_objs:
                    patterns.append({
                        'type': 'hidden_connection',
                        'entities': [node_a, node_c],
                        'intermediary': suspicious_objs,
                        'description': f"Suspicious link: {node_a} and {node_c} both interacted with {suspicious_objs}"
                    })
        
        
        # 模式3：空间矛盾
        for node in all_nodes:
            locations = []
            for u, v, data in self.graph.edges(node, data=True):
                # 1. 获取原始时间戳并统一转为列表处理
                raw_time = data.get('relation_timestamp', [])
                if not isinstance(raw_time, list):
                    raw_time = [raw_time]
                    
                # 2. 检查源节点是否为位置
                if self.graph.nodes[u].get('type', 'none') == 'location':
                    for t in raw_time:
                        locations.append({
                            'time': float(t), # 确保是数值类型
                            'place': u,
                            'turn': data.get('turn_id', 0)
                        })
                
                # 3. 检查目标节点是否为位置
                if self.graph.nodes[v].get('type', 'none') == 'location':
                    for t in raw_time:
                        locations.append({
                            'time': float(t),
                            'place': v,
                            'turn': data.get('turn_id', 0)
                        })
            
            # 4. 按时间点从小到大排序
            locations.sort(key=lambda x: x['time'])
            
            # 5. 检测短时间内位置的突变
            for i in range(len(locations) - 1):
                loc1 = locations[i]
                loc2 = locations[i+1]
                
                time_diff = loc2['time'] - loc1['time']
                
                # 如果时间间隔小于阈值（且不为0），但地点不同，则视为瞬移异常
                if 0 < time_diff < threshold_seconds and loc1['place'] != loc2['place']:
                    patterns.append({
                        'type': 'teleportation_anomaly',
                        'description': f"Entity {node} moved from {loc1['place']} to {loc2['place']} in {time_diff:.2f}s",
                        'evidence': [loc1, loc2]
                    })

        return patterns
    
    def get_reasoning_path(self, question, candidates):
        """
        根据提问词选择最优推理路径：
        - Who/What -> 实体特征匹配路径
        - Why -> 因果链（查找动作前后的状态变化）
        - When -> 时序排序路径
        """
        question = question.lower()
        
        # 1. Who 问题：侧重于从 Object/Action 溯源到 Person
        if any(w in question for w in ["who", "whose", "which person"]):
            return {"strategy": "actor_retrieval", "data": self.query_evidence_chains(question, candidates, None)}
        
        # 2. Why 问题：侧重于查找前序事件 (Pre-conditions)
        elif "why" in question:
            # 查找图中所有的 edge，寻找带有因果语义的 relation_description
            causal_links = [f"{u}->{d.get('type', 'none')}->{v}: {d.get('relation_description', 'none')}" for u,v,d in self.graph.edges(data=True) 
                           if "because" in d.get('relation_description', 'none').lower()]
            return {"strategy": "causal_inference", "evidence": causal_links}
        
        # 3. When 问题：返回完整的结构化时间线
        elif "when" in question or "before" in question or "after" in question:
            return {"strategy": "temporal_sorting", "timeline": self.get_timeline()}
        
        return {"strategy": "general_summary", "content": self.extract_complete_evidence()}
    
    def extract_complete_evidence(self):
        """
        导出完整的证据结构（用于最终答案生成）
        """
        evidence = {
            'summary':self.summarize(),
            'timeline': self.get_timeline(),
            'suspicious_patterns': self.find_suspicious_patterns(),
            'key_entities': {n: d for n, d in self.graph.nodes(data=True)},
            'relational_map': [f"{u}-{d.get('type', 'none')}->{v}" for u, v, d in self.graph.edges(data=True)]
        }
        return evidence
    
    def get_timeline(self):
        """
        生成时间线
        """
        events = []
        # 提取所有带有时间戳的边
        for u, v, data in self.graph.edges(data=True):
            events.append({
                'timestamp': data.get('relation_timestamp', []),
                'description': f"[{data.get('relation_timestamp', []):.1f}s] {u}-{data.get('type', 'none')}->{v}"
            })
        
        # 提取所有节点第一次出现的时刻
        for node, data in self.graph.nodes(data=True):
            events.append({
                'timestamp': data.get('first_seen', 0),
                'description': f"[{data.get('first_seen', 0):.1f}s] {node} ({data.get('type', 'none')}) first appeared in scene."
            })
            
        # 全局排序
        events.sort(key=lambda x: x['timestamp'])
        return events
    




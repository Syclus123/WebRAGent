import json
import pickle
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
import os
from pathlib import Path
import faiss
from dataclasses import dataclass
from PIL import Image
import ast

from src.arguments import ModelArguments, DataArguments, TrainingArguments
try:
    from transformers.hf_argparser import HfArgumentParser
    from transformers.models.auto.configuration_auto import AutoConfig
except ImportError:
    from transformers import HfArgumentParser, AutoConfig

from src.model.model import MMEBModel
from src.model.processor import get_backbone_name, load_processor
from src.data.collator.eval_collator import MultimodalEvalDataCollator


@dataclass
class QueryItem:
    """query item data structure"""
    text: str
    image_paths: Optional[List[str]] = None
    
    
@dataclass
class RetrievalResult:
    """retrieval result data structure"""
    cand_id: str
    score: float
    cand_text: str
    cand_image_paths: List[str]
    task_description: str
    annotation_id: str


class MultimodalRAGDatabase:
    """multimodal RAG database, support text and image joint retrieval"""
    
    def __init__(self, 
                 model_args: ModelArguments,
                 data_args: DataArguments, 
                 training_args: TrainingArguments,
                 cand_json_path: str,
                 embedding_parquet_path: str,
                 device: str = "cuda",
                 normalize_embeddings: bool = True):
        """
        Initialize the RAG database

        Args:
            model_args: Model parameters
            data_args: Data parameters
            training_args: The training parameters
            cand_json_path: The candidate JSON file path
            embedding_parquet_path: embedding vector parquet file path
            device: The computing device
            normalize_embeddings: Whether to normalize vector embeddings
        """
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        
        self._load_model_and_processor()
        
        # load data
        self.cand_data = self._load_candidate_data(cand_json_path)
        self.embedding_data = self._load_embedding_data(embedding_parquet_path)
        
        # build FAISS index
        self.index, self.id_map = self._build_faiss_index()
        
        print(f"RAG database initialized:")
        print(f"- candidate task number: {len(self.cand_data)}")
        print(f"- embedding vector number: {len(self.embedding_data)}")
        print(f"- FAISS index dimension: {self.index.d}")
        
    def _load_model_and_processor(self):
        """load model and processor"""
        print("loading model and processor...")
        
        # load config
        hf_config = AutoConfig.from_pretrained(self.model_args.model_name, trust_remote_code=True)
        if not hasattr(self.model_args, "model_backbone") or not self.model_args.model_backbone:
            model_backbone = get_backbone_name(hf_config=hf_config, model_type=self.model_args.model_type)
            setattr(self.model_args, 'model_backbone', model_backbone)
            setattr(self.training_args, 'model_backbone', model_backbone)
            
        # load processor and model
        self.processor = load_processor(self.model_args, self.data_args)
        self.model = MMEBModel.load(self.model_args, is_trainable=False)
        self.model.eval()
        self.model = self.model.to(self.device, dtype=torch.bfloat16)
        
        # create data collator
        self.collator = MultimodalEvalDataCollator(self.processor, self.model_args, self.data_args, "cand")
        
        print("model and processor loaded")
        
    def _load_candidate_data(self, json_path: str) -> Dict:
        """load candidate task data"""
        print(f"loading candidate task data: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON file contains invalid nested JSON string: {e}")
        
        # process different data formats
        if isinstance(data, dict):
            print("detected dictionary format JSON data")
            cand_dict = {}
            
            for key, item in data.items():
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError:
                        print(f"warning: skip invalid JSON string item with key '{key}'")
                        continue
                
                # ensure item is a dictionary
                if not isinstance(item, dict):
                    print(f"warning: skip non-dictionary item with key '{key}', type: {type(item)}")
                    continue
                
                # if item has no cand_id field, use key as cand_id
                if 'cand_id' not in item:
                    item['cand_id'] = key
                
                cand_id = item['cand_id']
                
                # check required fields
                required_fields = ['cand_text', 'cand_image_path', 'task']
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    print(f"warning: skip cand_id '{cand_id}', missing fields: {missing_fields}")
                    continue
                
                cand_dict[cand_id] = {
                    'cand_text': item['cand_text'],
                    'cand_image_path': item['cand_image_path'],
                    'task': item['task']
                }
                
        elif isinstance(data, list):
            print("detected list format JSON data")
            cand_dict = {}
            
            for i, item in enumerate(data):
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError:
                        print(f"warning: skip invalid JSON string item at index {i}")
                        continue
                
                # ensure item is a dictionary
                if not isinstance(item, dict):
                    print(f"warning: skip non-dictionary item at index {i}, type: {type(item)}")
                    continue
                
                # check required fields
                required_fields = ['cand_id', 'cand_text', 'cand_image_path', 'task']
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    print(f"warning: skip item at index {i}, missing fields: {missing_fields}")
                    continue
                
                cand_id = item['cand_id']
                cand_dict[cand_id] = {
                    'cand_text': item['cand_text'],
                    'cand_image_path': item['cand_image_path'],
                    'task': item['task']
                }
        else:
            raise ValueError(f"JSON file should contain a list or dictionary, but got {type(data)}")
        
        print(f"candidate task data loaded, {len(cand_dict)} tasks")
        return cand_dict
        
    def _load_embedding_data(self, parquet_path: str) -> pd.DataFrame:
        """load embedding data"""
        print(f"loading embedding data: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"embedding data loaded, {len(df)} vectors")
        return df
        
    def _build_faiss_index(self) -> Tuple[faiss.Index, List[str]]:
        """build FAISS index"""
        print("building FAISS index...")
        
        # extract embedding vectors
        embeddings = []
        id_map = []
        
        for _, row in self.embedding_data.iterrows():
            cand_id = row['cand_id']
            embed = row['embed']
            
            if isinstance(embed, (list, tuple)):
                embed = np.array(embed)
            elif isinstance(embed, str):
                try:
                    embed = np.array(ast.literal_eval(embed))
                except:
                    print(f"warning: cannot parse embedding {cand_id}")
                    continue
                    
            embeddings.append(embed)
            id_map.append(cand_id)
        
        embeddings = np.array(embeddings).astype(np.float32)
        
        # normalize embedding
        if self.normalize_embeddings:
            faiss.normalize_L2(embeddings)
            
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # inner product similarity
        index.add(embeddings)
        
        print(f"FAISS index built, dimension: {dimension}, vector number: {len(embeddings)}")
        return index, id_map
        
    def encode_query(self, query: QueryItem) -> np.ndarray:
        """encode query text and image to embedding vector"""
        
        # import necessary tokens
        from src.model.processor import VLM_IMAGE_TOKENS, QWEN2_VL_TOKENSELECTION
        
        # prepare query text, if there are images, add image tokens
        query_text = query.text
        if query.image_paths and len(query.image_paths) > 0:
            # add image tokens for each image
            image_token = VLM_IMAGE_TOKENS.get(QWEN2_VL_TOKENSELECTION, "<|image_pad|>")
            image_tokens = " ".join([image_token] * len(query.image_paths))
            query_text = f"{image_tokens} {query_text}"
        
        # build input data structure
        image_paths = query.image_paths or []
        query_data = {
            'cand_text': [query_text],
            'cand_image': [{"bytes": [None] * len(image_paths), "paths": image_paths, "resolutions": [None] * len(image_paths)}],
            'dataset_infos': [{"cand_id": "query", "retrieval_type": None}]
        }
        
        # use collator to process data
        batch = self.collator([query_data])
        inputs, _ = batch
        
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                output = self.model(tgt=inputs)
                embedding = output["tgt_reps"].cpu().detach().float().numpy()
        
        if self.normalize_embeddings:
            faiss.normalize_L2(embedding)
            
        return embedding[0]
        
    def search(self, 
               query: QueryItem, 
               top_k: int = 5,
               score_threshold: float = 0.0) -> List[RetrievalResult]:
        """
        Retrieving similarity tasks

        Args:
            query: The query item (contains text and optionally an image)
            top_k: The number of results returned
            score_threshold: The score threshold below which results will be filtered

        Returns:
            Retrieving a list of results
        """
        query_embedding = self.encode_query(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < score_threshold:
                continue
                
            cand_id = self.id_map[idx]
            
            # get embedding data info
            embed_row = self.embedding_data[self.embedding_data['cand_id'] == cand_id].iloc[0]
            annotation_id = embed_row['annotation_id']
            instruction = embed_row.get('instruction', '')
            
            # get candidate task info
            if cand_id in self.cand_data:
                cand_info = self.cand_data[cand_id]
                cand_text = cand_info['cand_text']
                cand_image_paths = cand_info['cand_image_path']
                task_description = cand_info['task']
            else:
                cand_text = ""
                cand_image_paths = []
                task_description = instruction
            
            result = RetrievalResult(
                cand_id=cand_id,
                score=float(score),
                cand_text=cand_text,
                cand_image_paths=cand_image_paths,
                task_description=task_description,
                annotation_id=annotation_id
            )
            results.append(result)
            
        return results
        
    def get_candidate_info(self, cand_id: str) -> Optional[Dict]:
        """get candidate task info by cand_id"""
        if cand_id in self.cand_data:
            return self.cand_data[cand_id]
        return None
        
    def save_index(self, index_path: str):
        """save FAISS index to file"""
        faiss.write_index(self.index, index_path)
        
        id_map_path = index_path.replace('.index', '_id_map.pkl')
        with open(id_map_path, 'wb') as f:
            pickle.dump(self.id_map, f)
            
        print(f"index saved to {index_path}")
        print(f"id map saved to {id_map_path}")
        
    def load_index(self, index_path: str):
        """load FAISS index from file"""
        self.index = faiss.read_index(index_path)
        
        id_map_path = index_path.replace('.index', '_id_map.pkl')
        with open(id_map_path, 'rb') as f:
            self.id_map = pickle.load(f)
            
        print(f"index loaded from {index_path}")
        print(f"id map loaded from {id_map_path}")


def create_rag_database_from_config(
    model_name: str,
    checkpoint_path: str,
    model_backbone: str,
    cand_json_path: str,
    embedding_parquet_path: str,
    device: str = "cuda",
    **kwargs
) -> MultimodalRAGDatabase:
    """
    Convenience function to create RAG database from configuration

    Args:
        model_name: The model name path
        checkpoint_path: The checkpoint path
        model_backbone: The model backbone network type
        cand_json_path: The candidate JSON file path
        embedding_parquet_path: The embedding vector file path
        device: The computing device
        **kwargs: Other parameters

    Returns:
        Initialized RAG database instance
    """
    
    model_args = ModelArguments(
        model_name=model_name,
        model_backbone=model_backbone,
        checkpoint_path=checkpoint_path,
        lora=kwargs.get('lora', True),
        pooling=kwargs.get('pooling', 'eos'),
        normalize=kwargs.get('normalize', True),
        **{k: v for k, v in kwargs.items() if k.startswith('model_')}
    )
    
    data_args = DataArguments(
        resize_use_processor=kwargs.get('resize_use_processor', True),
        max_len=kwargs.get('max_len', 65536),
        **{k: v for k, v in kwargs.items() if k.startswith('data_')}
    )
    
    training_args = TrainingArguments(
        # device=device,
        per_device_eval_batch_size=kwargs.get('per_device_eval_batch_size', 1),
        dataloader_num_workers=kwargs.get('dataloader_num_workers', 2),
        **{k: v for k, v in kwargs.items() if k.startswith('training_')}
    )
    
    # create RAG database
    rag_db = MultimodalRAGDatabase(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        cand_json_path=cand_json_path,
        embedding_parquet_path=embedding_parquet_path,
        device=device,
        normalize_embeddings=kwargs.get('normalize_embeddings', True)
    )
    
    return rag_db


if __name__ == "__main__":
    rag_db = create_rag_database_from_config(
        model_name="/home/ubuntu/data/csb/Embedding/Qwen2-VL-TokenSelection-2B",
        checkpoint_path="/home/ubuntu/data/csb/Embedding/experiments/train/qwen2_vl-lite_full-lora8-bsz128x8x2-interleave_0.2-lr5e5-max_step_256-warmup_12-uigraph_select_0.5-lm_skip_all-vis_skip_all/huggingface",
        model_backbone="qwen2_vl_tokenselection",
        cand_json_path="/path/to/cand_with_task.json",
        embedding_parquet_path="/path/to/trajectory_embedding.parquet",
        lora=True,
        pooling="eos",
        normalize=True,
        resize_use_processor=True,
        max_len=65536
    )
    
    query = QueryItem(
        text="open the website and click the login button",
        image_paths=["/path/to/screenshot.jpg"]
    )
    
    results = rag_db.search(query, top_k=5)
    
    for i, result in enumerate(results):
        print(f"result {i+1}:")
        print(f"  - cand_id: {result.cand_id}")
        print(f"  - similarity score: {result.score:.4f}")
        print(f"  - task description: {result.task_description}")
        print(f"  - candidate text: {result.cand_text[:200]}...")
        print()
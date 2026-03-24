#!/usr/bin/env python3
"""
GPT-4 Re-ranking Based Image Retrieval Evaluation Script (test)

usage:
    python batch_eval_img_rerank.py --optimized-gpt4-test
"""

import os
import sys
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import datetime
import time
import numpy as np
import base64
from openai import OpenAI
from PIL import Image
import io
import hashlib
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from rag_database import create_rag_database_from_config, QueryItem, MultimodalRAGDatabase

# GPT-4
GPT4_CLIENT = None
GPT4_MODEL = "gpt-4.1"
ENABLE_GPT4_LOG = True
GPT4_LOG_FILE = None

# cache config
CACHE_DIR = "/tmp/gpt4_cache"
ENABLE_CACHE = True

def get_cache_key(query_image_path: str, query_task_description: str, candidate_ids: List[str]) -> str:
    """generate cache key"""
    content = f"{query_image_path}_{query_task_description}_{'-'.join(candidate_ids[:5])}"
    return hashlib.md5(content.encode()).hexdigest()

def load_from_cache(cache_key: str) -> Optional[Tuple[str, str, float]]:
    """load result from cache"""
    if not ENABLE_CACHE:
        return None
    
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"cache loading failed: {e}")
    
    return None

def save_to_cache(cache_key: str, result: Tuple[str, str, float]) -> None:
    """save result to cache"""
    if not ENABLE_CACHE:
        return
    
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    except Exception as e:
        print(f"cache saving failed: {e}")

# new method for image encoding
def encode_image_to_base64_compressed(image_path_or_base64: str) -> str:
    """encode image to base64 format with advanced compression for token efficiency
    Supports both file paths and base64 inputs, with size scaling and quality compression
    """
    try:
        # if already base64, return directly
        if image_path_or_base64.startswith('data:image'):
            return image_path_or_base64.split(',')[1]
        elif len(image_path_or_base64) > 100 and '/' not in image_path_or_base64:
            # Already base64 encoded string
            return image_path_or_base64
        
        import base64
        import os
        from PIL import Image
        import io
        
        if not os.path.exists(image_path_or_base64):
            print(f"⚠️  image file not found: {image_path_or_base64}")
            return ""
        
        # read and compress image for token efficiency
        with Image.open(image_path_or_base64) as img:
            max_size = 600
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # convert to base64 with lower quality
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG", quality=60, optimize=True)  # Enhanced with optimize=True
            img_bytes = img_byte_arr.getvalue()
            
            encoded = base64.b64encode(img_bytes).decode('utf-8')
            print(f"📸 Image compressed: {len(encoded)//4} tokens (size: {img.width}x{img.height})")
            return encoded
            
    except Exception as e:
        print(f"❌ image encoding failed {image_path_or_base64}: {e}")
        return ""
    
    
# def encode_image_to_base64_compressed(image_path: str) -> str:
#     """encode image to base64 format, and compress it"""
#     try:
#         with open(image_path, "rb") as image_file:
#             # read original image data
#             image_data = image_file.read()
            
#             # use Pillow to compress
#             compressed_image_data = io.BytesIO()
            
#             # open image and compress
#             img = Image.open(io.BytesIO(image_data))
#             img.save(compressed_image_data, format="JPEG", quality=85) # use JPEG format, quality 85
            
#             compressed_image_data.seek(0)
#             compressed_image_bytes = compressed_image_data.getvalue()
            
#             return base64.b64encode(compressed_image_bytes).decode('utf-8')
#     except Exception as e:
#         print(f"image compression encoding failed {image_path}: {e}")
#         return ""

@lru_cache(maxsize=1000)
def get_compressed_image_base64(image_path: str) -> str:
    return encode_image_to_base64_compressed(image_path)

def initialize_gpt4_client(api_key: Optional[str] = None):
    """initialize GPT-4 client"""
    global GPT4_CLIENT
    
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
        
    if api_key is None:
        print("warning: OpenAI API key not found, Plan2 method will be unavailable")
        print("please set environment variable OPENAI_API_KEY or provide API key in the code")
        return False
    
    try:
        GPT4_CLIENT = OpenAI(api_key=api_key)
        print("GPT-4 ok")
        return True
    except Exception as e:
        print(f"GPT-4 client initialization failed: {e}")
        return False

def gpt4_rerank_results(query_image_path: str, query_task_description: str, search_results: List[Dict], top_k: int = 15, cand_img_num: int = 15) -> Tuple[str, str, float]:
    """
    GPT-4 was used to reorder the retrieved results

    Args:
        query_image_path: Finds the image path
        query_task_description: Query the task description
        search_results: A list of results, each containing cand_id, task_description, candidate image path, etc
        top_k: The top K results considered (default 15)
        cand_img_num: The number of candidate images to send to GPT-4 (default 8)
        
    Returns:
        (best_task_id, best_cand_id, fixed_confidence)
    """
    global GPT4_CLIENT, ENABLE_GPT4_LOG, GPT4_LOG_FILE
    
    if GPT4_CLIENT is None:
        print("GPT-4 client not initialized, return original Top1 result")
        if search_results:
            cand_id = search_results[0]['cand_id']
            task_id = cand_id.split('_traj-')[0] if '_traj-' in cand_id else cand_id
            return task_id, cand_id, 1.0
        return "", "", 1.0
    
    if not search_results:
        return "", "", 1.0
    
    # check cache
    candidate_ids = [r.get('cand_id', '') for r in search_results[:top_k]]
    cache_key = get_cache_key(query_image_path, query_task_description, candidate_ids)
    cached_result = load_from_cache(cache_key)
    if cached_result:
        return cached_result
    
    # encode query image (using compression)
    query_image_base64 = get_compressed_image_base64(query_image_path)
    if not query_image_base64:
        return "", "", 0.0
    
    # prepare candidate description (only process the first top_k, default 15)
    candidates_text = []
    candidate_images = []
    
    for i, result in enumerate(search_results[:top_k]):
        task_desc = result.get('task_description', '无任务描述')
        cand_id = result.get('cand_id', f'candidate_{i}')
        
        if '_traj-' in cand_id:
            parts = cand_id.split('_traj-')
            task_hash = parts[0]
            step_num = parts[1]
            candidate_image_path = f"/home/ubuntu/data/csb/images/embedding/GAE-Bench/images/Online-Mind2Web/{task_hash}_step_{step_num}.png"
        else:
            candidate_image_path = ""
        
        # detailed candidate information
        candidate_info = f"""
Candidate {i+1}:
- Candidate ID: {cand_id}
- Task Description: {task_desc}
"""
        candidates_text.append(candidate_info)
        candidate_images.append(candidate_image_path)
    
    prompt = f"""
You are a web interface image matching expert. I will provide you with one query web interface image, a query task description, and {len(candidates_text)} candidate web interfaces, each with corresponding web interface images and task descriptions.

Query Task Description: {query_task_description}

Please carefully analyze the query image and candidate images, and match them based on task descriptions, including:
- Interface elements (buttons, input fields, text, etc.)
- Interface layout and design
- Interface visual similarity
- Matching degree between query task and candidate task descriptions

Candidate Information:
{"".join(candidates_text)}

Please select the most matching candidate from the above options, considering the following main factors:
1. Image visual similarity
2. Interface element matching degree
3. Relevance between query task and candidate task descriptions

Please answer:
The most matching candidate number (1-{len(candidates_text)})

Answer format:
Best matching candidate: [number]
"""
    
    try:
        message_content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "\nQuery Image:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{query_image_base64}",
                    "detail": "low"
                }
            }
        ]
        
        for i, candidate_image_path in enumerate(candidate_images[:cand_img_num]):
            if candidate_image_path and os.path.exists(candidate_image_path):
                candidate_image_base64 = get_compressed_image_base64(candidate_image_path)
                if candidate_image_base64:
                    message_content.extend([
                        {"type": "text", "text": f"\nCandidate {i+1} Image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{candidate_image_base64}",
                                "detail": "low"
                            }
                        }
                    ])
        
        request_data = {
            "model": GPT4_MODEL,
            "messages": [{"role": "user", "content": message_content}],
            "max_tokens": 1500,  
            "temperature": 0.0 
        }
        
        if ENABLE_GPT4_LOG and GPT4_LOG_FILE:
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "query_image_path": query_image_path,
                "candidates_count": len(search_results[:top_k]),
                "image_parts_sent": len([item for item in message_content if item["type"] == "image_url"]),
                "optimized_version": True
            }
            
            gpt4_logs = []
            if os.path.exists(GPT4_LOG_FILE):
                try:
                    with open(GPT4_LOG_FILE, 'r', encoding='utf-8') as f:
                        gpt4_logs = json.load(f)
                except:
                    gpt4_logs = []
            
            gpt4_logs.append(log_entry)
            
            with open(GPT4_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(gpt4_logs, f, ensure_ascii=False, indent=2)
        
        response = GPT4_CLIENT.chat.completions.create(**request_data)
        response_text = response.choices[0].message.content
        
        best_index = 0
        confidence = 1.0
        
        if response_text:
            # parse English answer format
            # find "Best matching candidate:" 
            candidate_match = re.search(r'Best matching candidate[:：]\s*(\d+)', response_text, re.IGNORECASE)
            if candidate_match:
                best_index = int(candidate_match.group(1)) - 1
            else:
                chinese_match = re.search(r'最匹配候选项[:：]\s*(\d+)', response_text)
                if chinese_match:
                    best_index = int(chinese_match.group(1)) - 1
                else:
                    numbers = re.findall(r'\b([1-9]\d?)\b', response_text)
                    if numbers:
                        best_index = int(numbers[0]) - 1
        
        # ensure index is valid
        best_index = max(0, min(best_index, len(search_results) - 1))
        
        if best_index < len(search_results):
            best_result = search_results[best_index]
            best_cand_id = best_result['cand_id']
            best_task_id = best_cand_id.split('_traj-')[0] if '_traj-' in best_cand_id else best_cand_id
            result = (best_task_id, best_cand_id, confidence)
            
            save_to_cache(cache_key, result)
            return result
        
    except Exception as e:
        print(f"GPT-4 reranking error: {e}")
    
    # return original Top1 result when error
    if search_results:
        cand_id = search_results[0]['cand_id']
        task_id = cand_id.split('_traj-')[0] if '_traj-' in cand_id else cand_id
        return task_id, cand_id, 1.0
    
    return "", "", 1.0


@dataclass
class TestSample:
    """test sample data structure"""
    image_path: str
    task_id: str
    step_id: int
    expected_cand_id: str
    task_description: str = ""


class GPT4RetrievalMetrics:
    """GPT-4 metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.task_top1_correct = 0
        self.step_top1_correct = 0
        self.gpt4_task_correct = 0
        self.gpt4_step_correct = 0
        self.total_samples = 0
        self.total_search_time = 0.0
        self.total_gpt4_time = 0.0
        self.search_times = []
        self.gpt4_times = []
        self.gpt4_confidences = []
    
    def update(self, expected_cand_id: str, predicted_cand_ids: List[str], search_time: float = 0.0, 
               gpt4_task_id: str = "", gpt4_cand_id: str = "", gpt4_time: float = 0.0, gpt4_confidence: float = 0.0):
        self.total_samples += 1
        self.total_search_time += search_time
        self.total_gpt4_time += gpt4_time
        self.search_times.append(search_time)
        self.gpt4_times.append(gpt4_time)
        self.gpt4_confidences.append(gpt4_confidence)
        
        # extract expected task ID
        expected_task_id = expected_cand_id.split('_traj-')[0] if '_traj-' in expected_cand_id else expected_cand_id
        
        # original Top1 accuracy
        if predicted_cand_ids:
            predicted_task_id = predicted_cand_ids[0].split('_traj-')[0] if '_traj-' in predicted_cand_ids[0] else predicted_cand_ids[0]
            
            if predicted_task_id == expected_task_id:
                self.task_top1_correct += 1
            if predicted_cand_ids[0] == expected_cand_id:
                self.step_top1_correct += 1
        
        # GPT-4 reranking accuracy
        if gpt4_task_id == expected_task_id:
            self.gpt4_task_correct += 1
        if gpt4_cand_id == expected_cand_id:
            self.gpt4_step_correct += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """get evaluation results"""
        if self.total_samples == 0:
            return {
                "task_top1_accuracy": 0.0,
                "step_top1_accuracy": 0.0,
                "gpt4_task_accuracy": 0.0,
                "gpt4_step_accuracy": 0.0,
                "total_samples": 0.0,
                "avg_search_time": 0.0,
                "total_search_time": 0.0,
                "avg_gpt4_time": 0.0,
                "total_gpt4_time": 0.0,
                "avg_gpt4_confidence": 0.0,
                "median_search_time": 0.0,
                "min_search_time": 0.0,
                "max_search_time": 0.0
            }
        
        return {
            "task_top1_accuracy": float(self.task_top1_correct / self.total_samples),
            "step_top1_accuracy": float(self.step_top1_correct / self.total_samples),
            "gpt4_task_accuracy": float(self.gpt4_task_correct / self.total_samples),
            "gpt4_step_accuracy": float(self.gpt4_step_correct / self.total_samples),
            "total_samples": float(self.total_samples),
            "avg_search_time": float(self.total_search_time / self.total_samples),
            "total_search_time": float(self.total_search_time),
            "avg_gpt4_time": float(self.total_gpt4_time / self.total_samples),
            "total_gpt4_time": float(self.total_gpt4_time),
            "avg_gpt4_confidence": float(np.mean(self.gpt4_confidences)) if self.gpt4_confidences else 0.0,
            "median_search_time": float(np.median(self.search_times)) if self.search_times else 0.0,
            "min_search_time": float(min(self.search_times)) if self.search_times else 0.0,
            "max_search_time": float(max(self.search_times)) if self.search_times else 0.0
        }


def load_test_samples(image_dir: str, cand_data: Dict, num_samples: int = 50, random_seed: int = 42) -> List[TestSample]:
    """randomly select test samples from image directory"""
    
    random.seed(random_seed)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') and '_step_' in f]
    
    print(f"find {len(image_files)} images")
    
    # randomly select specified number of images
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    test_samples = []
    
    for image_file in selected_files:
        # parse file name: 0b838cd54f826c59c71f600c56b89a11_step_3.png
        # corresponding task: 0b838cd54f826c59c71f600c56b89a11_traj-3
        
        if '_step_' not in image_file:
            continue
            
        base_name = image_file.replace('.png', '')
        parts = base_name.split('_step_')
        
        if len(parts) != 2:
            continue
            
        task_hash = parts[0]
        step_num = parts[1]
        
        # construct expected cand_id
        expected_cand_id = f"{task_hash}_traj-{step_num}"
        
        # find corresponding task description
        task_description = ""
        if expected_cand_id in cand_data:
            task_description = cand_data[expected_cand_id].get('task', '')
        else:
            for cand_id in cand_data:
                if cand_id.startswith(task_hash + '_traj-'):
                    task_description = cand_data[cand_id].get('task', '')
                    if task_description:
                        break
        
        image_path = os.path.join(image_dir, image_file)
        if not os.path.exists(image_path):
            print(f"warning: image file not found, skip: {image_path}")
            continue
        
        test_sample = TestSample(
            image_path=image_path,
            task_id=task_hash,
            step_id=int(step_num),
            expected_cand_id=expected_cand_id,
            task_description=task_description
        )
        
        test_samples.append(test_sample)
    
    print(f"successfully create {len(test_samples)} test samples")
    return test_samples


def image_only_retrieval(sample: TestSample, rag_db: MultimodalRAGDatabase, top_k: int = 20) -> Tuple[List[str], List[float], List[Dict], float]:
    """optimized image retrieval method"""
    
    start_time = time.time()
    
    # create pure image query
    query = QueryItem(
        text="",  # empty text
        image_paths=[sample.image_path] if os.path.exists(sample.image_path) else []
    )
    
    # execute search
    results = rag_db.search(query, top_k=top_k, score_threshold=0.0)
    
    search_time = time.time() - start_time
    
    return (
        [result.cand_id for result in results],
        [result.score for result in results],
        [{
            'cand_id': result.cand_id,
            'score': result.score,
            'task_description': result.task_description,
            'cand_text': result.cand_text,
            'annotation_id': result.annotation_id
        } for result in results],
        search_time
    )


def run_gpt4_rerank_evaluation(config: Dict, num_samples: int = 50, verbose: bool = True, cand_img_num: int = 8):
    """run GPT-4 reranking image retrieval evaluation"""
    
    print("=" * 80)
    print("GPT-4 reranking image retrieval method evaluation")
    print("=" * 80)
    print(f"number of samples: {num_samples}")
    print(f"number of candidate images: {cand_img_num}")
    print(f"cache status: {'enabled' if ENABLE_CACHE else 'disabled'}")
    
    # initialize GPT-4 client
    print("initializing GPT-4 client...")
    gpt4_available = initialize_gpt4_client()
    if not gpt4_available:
        print("GPT-4 client initialization failed, cannot evaluate")
        return None
    
    print("initializing RAG database...")
    start_init_time = time.time()
    rag_db = create_rag_database_from_config(**config)
    init_time = time.time() - start_init_time
    print(f"RAG database initialization completed, time: {init_time:.2f} seconds")
    
    # save FAISS index (if specified)
    if 'index_save_path' in config:
        print(f"saving FAISS index to: {config['index_save_path']}")
        rag_db.save_index(config['index_save_path'])
    
    # test image directory
    image_dir = "/home/ubuntu/data/csb/images/embedding/GAE-Bench/images/Online-Mind2Web"
    test_samples = load_test_samples(image_dir, rag_db.cand_data, num_samples=num_samples)
    
    if not test_samples:
        print("no valid test samples found, exit")
        return None
    
    # initialize evaluation metrics
    metrics = GPT4RetrievalMetrics()
    
    print(f"\n start GPT-4 reranking retrieval evaluation, total {len(test_samples)} samples")
    print("=" * 80)
    
    # detailed results
    detailed_results = []
    
    for i, sample in enumerate(test_samples, 1):
        if verbose:
            print(f"\nprocessing sample {i}/{len(test_samples)}: {sample.expected_cand_id}")
            print(f"image path: {sample.image_path}")
            print(f"task description: {sample.task_description[:100]}...")
        
        if not os.path.exists(sample.image_path):
            print(f" warning: image not found, skip")
            continue
        
        # pure image retrieval
        try:
            pred_ids, scores, details, search_time = image_only_retrieval(sample, rag_db, top_k=20)
            
            # GPT-4 reranking
            gpt4_task_id, gpt4_cand_id, gpt4_confidence, gpt4_time = "", "", 0.0, 0.0
            if details:
                start_gpt4_time = time.time()
                try:
                    gpt4_task_id, gpt4_cand_id, gpt4_confidence = gpt4_rerank_results(
                        sample.image_path, sample.task_description, details[:15], top_k=15, cand_img_num=cand_img_num
                    )
                    gpt4_time = time.time() - start_gpt4_time
                except Exception as e:
                    if verbose:
                        print(f"  GPT-4 reranking failed: {e}")
                    gpt4_time = time.time() - start_gpt4_time
            
            metrics.update(sample.expected_cand_id, pred_ids, search_time, 
                         gpt4_task_id, gpt4_cand_id, gpt4_time, gpt4_confidence)
            
            if verbose:
                print(f"  Top5 prediction: {pred_ids[:5]}")
                print(f"  Top5 scores: {[f'{s:.6f}' for s in scores[:5]]}")
                print(f"  search time: {search_time:.4f} seconds")
                if pred_ids:
                    expected_task_id = sample.expected_cand_id.split('_traj-')[0] if '_traj-' in sample.expected_cand_id else sample.expected_cand_id
                    print(f"  original Top1 match: {'✓' if pred_ids[0] == sample.expected_cand_id else '✗'}")
                    
                    # GPT-4 reranking results
                    if gpt4_cand_id:
                        print(f"  GPT-4 task prediction: {gpt4_task_id} ({'✓' if gpt4_task_id == expected_task_id else '✗'})")
                        print(f"  GPT-4 step prediction: {gpt4_cand_id} ({'✓' if gpt4_cand_id == sample.expected_cand_id else '✗'})")
                        print(f"  GPT-4 processing time: {gpt4_time:.4f} seconds")
            
        except Exception as e:
            print(f"  image retrieval error: {e}")
            pred_ids, scores, details, search_time = [], [], [], 0.0
            gpt4_task_id, gpt4_cand_id, gpt4_confidence, gpt4_time = "", "", 0.0, 0.0
        
        expected_task_id = sample.expected_cand_id.split('_traj-')[0] if '_traj-' in sample.expected_cand_id else sample.expected_cand_id
        
        sample_result = {
            "sample_id": i,
            "expected_cand_id": sample.expected_cand_id,
            "expected_task_id": expected_task_id,
            "task_id": sample.task_id,
            "step_id": sample.step_id,
            "task_description": sample.task_description,
            "image_path": sample.image_path,
            "predictions": pred_ids[:20],
            "scores": scores[:20],
            "search_time": search_time,
            "top1_correct": pred_ids[0] == sample.expected_cand_id if pred_ids else False,
            "target_rank": pred_ids.index(sample.expected_cand_id) + 1 if sample.expected_cand_id in pred_ids else -1,
            "gpt4_task_prediction": gpt4_task_id,
            "gpt4_step_prediction": gpt4_cand_id,
            "gpt4_task_correct": gpt4_task_id == expected_task_id,
            "gpt4_step_correct": gpt4_cand_id == sample.expected_cand_id,
            "gpt4_confidence": gpt4_confidence,
            "gpt4_time": gpt4_time
        }
        
        detailed_results.append(sample_result)
        
        if not verbose and i % 10 == 0:
            print(f"processed {i}/{len(test_samples)} samples...")
    
    # output final evaluation results
    print("\n" + "=" * 80)
    print("GPT-4 reranking retrieval evaluation results")
    print("=" * 80)
    
    results = metrics.get_metrics()
    
    print(f"\naccuracy comparison:")
    print(f"  original task Top1 accuracy: {results['task_top1_accuracy']:.2%} ({results['task_top1_accuracy']*results['total_samples']:.0f}/{results['total_samples']:.0f})")
    print(f"  original step Top1 accuracy: {results['step_top1_accuracy']:.2%} ({results['step_top1_accuracy']*results['total_samples']:.0f}/{results['total_samples']:.0f})")
    print(f"  GPT-4 task accuracy: {results['gpt4_task_accuracy']:.2%} ({results['gpt4_task_accuracy']*results['total_samples']:.0f}/{results['total_samples']:.0f})")
    print(f"  GPT-4 step accuracy: {results['gpt4_step_accuracy']:.2%} ({results['gpt4_step_accuracy']*results['total_samples']:.0f}/{results['total_samples']:.0f})")
    
    print(f"\nimprovement:")
    print(f"  task accuracy improvement: {results['gpt4_task_accuracy'] - results['task_top1_accuracy']:.2%}")
    print(f"  step accuracy improvement: {results['gpt4_step_accuracy'] - results['step_top1_accuracy']:.2%}")
    
    print(f"\nperformance metrics:")
    print(f"  average search time: {results['avg_search_time']:.4f} seconds")
    print(f"  average GPT-4 processing time: {results['avg_gpt4_time']:.4f} seconds")
    print(f"  total processing time: {results['total_search_time'] + results['total_gpt4_time']:.2f} seconds")
    
    # analyze improvement
    gpt4_better = len([r for r in detailed_results if r['gpt4_step_correct'] and not r['top1_correct']])
    gpt4_worse = len([r for r in detailed_results if not r['gpt4_step_correct'] and r['top1_correct']])
    gpt4_same = len([r for r in detailed_results if r['gpt4_step_correct'] == r['top1_correct']])
    
    print(f"\nGPT-4 reranking effect analysis:")
    print(f"  GPT-4 better samples: {gpt4_better} ({gpt4_better/len(detailed_results):.1%})")
    print(f"  GPT-4 worse samples: {gpt4_worse} ({gpt4_worse/len(detailed_results):.1%})")
    print(f"  same samples: {gpt4_same} ({gpt4_same/len(detailed_results):.1%})")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./test_data/output_image_new/gpt4_rerank_evaluation_{timestamp}.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    result_data = {
        "evaluation_time": timestamp,
        "evaluation_type": "gpt4_rerank_image_retrieval",
        "config": config,
        "initialization_time": init_time,
        "num_samples": len(test_samples),
        "cand_img_num": cand_img_num,
        "metrics": results,
        "samples": detailed_results,
        "summary": {
            "improvement_analysis": {
                "gpt4_better_samples": gpt4_better,
                "gpt4_worse_samples": gpt4_worse,
                "gpt4_same_samples": gpt4_same,
                "task_improvement": results['gpt4_task_accuracy'] - results['task_top1_accuracy'],
                "step_improvement": results['gpt4_step_accuracy'] - results['step_top1_accuracy']
            }
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"\ndetailed results saved to: {output_file}")
    return results


def run_optimized_gpt4_test(enable_log: bool = False, num_samples: int = 10, cand_img_num: int = 8):
    """Run an optimized version of the GPT-4 reordering test

    Args:
        enable_log: Whether logging is enabled
        num_samples: The number of test samples
        cand_img_num: The number of candidate images to send to GPT-4 (default 8)
    """
    global ENABLE_GPT4_LOG, GPT4_LOG_FILE, ENABLE_CACHE
    
    print("running optimized GPT-4 reordering test...")
    print(f"number of samples: {num_samples}")
    print(f"number of candidate images: {cand_img_num}")
    print(f"cache status: {'enabled' if ENABLE_CACHE else 'disabled'}")
    
    if enable_log:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        GPT4_LOG_FILE = f"./test_data/output_image_new/gpt4_optimized_log_{timestamp}.json"
        ENABLE_GPT4_LOG = True
        
        os.makedirs(os.path.dirname(GPT4_LOG_FILE), exist_ok=True)
        
        print(f"GPT-4 logging enabled, log file: {GPT4_LOG_FILE}")
    else:
        ENABLE_GPT4_LOG = False
        GPT4_LOG_FILE = None
        print("GPT-4 logging disabled")
    
    config = {
        "model_name": "/home/ubuntu/data/csb/Embedding/Qwen2-VL-TokenSelection-2B",
        "checkpoint_path": "/home/ubuntu/data/csb/Embedding/experiments/train/qwen2_vl-lite_full-lora8-bsz128x8x2-interleave_0.2-lr5e5-max_step_256-warmup_12-uigraph_select_0.5-lm_skip_all-vis_skip_all/huggingface",
        "model_backbone": "qwen2_vl_tokenselection",
        "cand_json_path": "/home/ubuntu/data/csb/Embedding/data/cand_with_task.json",
        "embedding_parquet_path": "/home/ubuntu/data/csb/Embedding/data/trajectory_embedding.parquet",
        "index_save_path": f"./test_data/output_image_new/faiss_index_optimized_{num_samples}.index",
        "lora": True,
        "pooling": "eos", 
        "normalize": True,
        "resize_use_processor": True,
        "max_len": 65536,
        "per_device_eval_batch_size": 2,
        "dataloader_num_workers": 2,
        "device": "cuda"
    }
    
    cache_clear_choice = input("clear GPT-4 cache to test optimization effect? (y/N): ").lower()
    if cache_clear_choice == 'y':
        try:
            import shutil
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
                print("GPT-4 cache cleaned")
        except Exception as e:
            print(f"cache cleaning failed: {e}")
    
    print("\nstart optimized test...")
    run_gpt4_rerank_evaluation(config, num_samples=num_samples, verbose=True, cand_img_num=cand_img_num)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--optimized-gpt4-test":
            enable_log = "--log" in sys.argv
            num_samples = 10 
            cand_img_num = 15 
            
            if "--num-samples" in sys.argv:
                try:
                    num_samples = int(sys.argv[sys.argv.index("--num-samples") + 1])
                except (ValueError, IndexError):
                    print("error: --num-samples parameter needs an integer")
                    sys.exit(1)
                    
            if "--cand_img" in sys.argv:
                try:
                    cand_img_num = int(sys.argv[sys.argv.index("--cand_img") + 1])
                except (ValueError, IndexError):
                    print("error: --cand_img parameter needs an integer")
                    sys.exit(1)
                    
            run_optimized_gpt4_test(enable_log=enable_log, num_samples=num_samples, cand_img_num=cand_img_num)
        else:
            print("  python batch_eval_img_rerank.py --optimized-gpt4-test [--num-samples <N>] [--cand_img <M>] [--log]")
            print("    --num-samples <N>: specify the number of test samples (default 10)")
            print("    --cand_img <M>: specify the number of candidate images (default 8)")
            print("    --log")
            print("")
            print("  python batch_eval_img_rerank.py --optimized-gpt4-test")
            print("  python batch_eval_img_rerank.py --optimized-gpt4-test --num-samples 30 --cand_img 15 --log")
    else:
        run_optimized_gpt4_test() 
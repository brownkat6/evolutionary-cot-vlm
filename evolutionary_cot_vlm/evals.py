from typing import List, Dict, Any, Optional, Tuple, Union
import json
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value, Sequence
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from rouge_score import rouge_scorer
import logging
from pathlib import Path
import os
import shutil
from utils.dataset_loading import get_chartqa_dataset
import parlai.core.build_data as build_data
from parlai.core.opt import Opt
from parlai.core.teachers import create_task_agent_from_taskname
from parlai.tasks.vqa_v2.agents import OeTeacher
from constants import CHARTQA_DIR, VQA_V2_DIR, MMMU_DIR, CACHE_DIR
import zipfile
import pickle
from functools import lru_cache
import base64
from transformers import LlavaProcessor, Blip2Processor
from lmms_eval.evaluator import evaluate
#from lmms_eval.evaluator import simple_evaluate, evaluate
from lmms_eval.tasks import TaskManager, get_task_dict
import lmms_eval
from models import simple_evaluate
import constants
from lmms_eval.api.model import lmms

import copy
import warnings
from typing import List, Optional, Tuple, Union

import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from tqdm import tqdm
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.tasks.mmmu.utils_group_img import process_images
from lmms_eval.utils import stop_sequences_criteria

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add module-level cache
_DATASET_CACHE: Dict[str, Dict[str, Any]] = {}

# Get cache settings from environment variables
_CACHE_DIR = Path(os.getenv('EVAL_CACHE_DIR', './cache'))
_USE_CACHE = os.getenv('USE_DATASET_CACHE', '1').lower() in ('1', 'true', 'yes')
_PRELOAD_IMAGES = os.getenv('PRELOAD_IMAGES', '1').lower() in ('1', 'true', 'yes')

# Create cache directory if it doesn't exist
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

class EvaluationError(Exception):
    """Custom exception for evaluation errors."""
    pass

class DatasetLoadError(Exception):
    """Custom exception for dataset loading errors."""
    pass

class MetricCalculationError(Exception):
    """Custom exception for metric calculation errors."""
    pass

class LMMWrapper(lmms):
    """
    Wrapper class to make models compatible with lmms-eval interface.
    Currently used for Molmo model which doesn't have native lmms-eval support.
    """
    
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.task_dict = {}
        self.rank = 0
        self.world_size = 1
        self.cache_requests = False
        self.rewrite_requests_cache = False
        self.apply_chat_template = False
        self.fewshot_as_multiturn = False

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            context = contexts[0]
            if "<image>" in context:
                # instruct blip does not expect the <image> tag
                context = context.replace("<image>", "")
            # Set trunction equals true here, the max length for qformer tokenizer is 512
            # if not truncate, some questions will cause size mismatch
            # The transformer implementation can't handle multi images for blip
            # Concat it into one image
            if len(visuals) > 1:
                visuals = [process_images(visuals)]
            inputs = self._image_processor(images=visuals, text=context, return_tensors="pt", truncation=True).to(self.device)

            gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            try:
                cont = self.model.generate(
                    **inputs,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = ""
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
            
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "We have not implemented this function for Molmo yet"

def download_chartqa(output_dir: Path = CHARTQA_DIR) -> None:
    """
    Download ChartQA dataset to specified directory.
    Default directory is CHARTQA_DIR from constants.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Downloading ChartQA Dataset to {output_dir} ===")
    
    # Download from HuggingFace
    url = "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip"
    zip_path = output_dir / "chartqa.zip"
    
    try:
        print("1. Downloading zip file...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc="Downloading"
        ) as pbar:
            for data in response.iter_content(8192):
                size = f.write(data)
                pbar.update(size)
        
        print(f"2. Extracting files... from {zip_path} to {output_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        print("3. Cleaning up...")
        zip_path.unlink()
        print(f"✓ ChartQA dataset successfully downloaded to {output_dir}")
        
    except Exception as e:
        print(f"! Error during download: {str(e)}")
        if zip_path.exists():
            zip_path.unlink()
        raise

def setup_vqa_v2(output_dir: Path) -> Path:
    """
    Setup VQA-v2 dataset using direct downloads if ParlAI fails.
    Returns the path to the dataset directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Setting up VQA-v2 Dataset ===")
    print(f"Target directory: {output_dir.absolute()}")
    
    # Define expected directory structure
    images_dir = output_dir / 'images'
    questions_dir = output_dir / 'questions'
    annotations_dir = output_dir / 'annotations'
    
    # Check if dataset is already set up properly
    expected_structure = {
        'images': {
            'train2014': images_dir / 'train2014',
            'val2014': images_dir / 'val2014',
            'test2015': images_dir / 'test2015'
        },
        'questions': {
            'train': questions_dir / 'v2_Questions_Train_mscoco.json',
            'val': questions_dir / 'v2_Questions_Val_mscoco.json',
            'test': questions_dir / 'v2_Questions_Test_mscoco.json'
        },
        'annotations': {
            'train': annotations_dir / 'v2_Annotations_Train_mscoco.json',
            'val': annotations_dir / 'v2_Annotations_Val_mscoco.json'
        }
    }
    
    # Check if already properly set up
    if all(path.exists() and (path.is_dir() and any(path.iterdir()) if 'images' in str(path) else path.is_file())
           for group in expected_structure.values() 
           for path in group.values()):
        print("✓ VQA-v2 dataset already properly set up")
        return output_dir
    
    # Define URLs for all splits
    urls = {
        # Images
        'train_images': 'http://images.cocodataset.org/zips/train2014.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2014.zip',
        'test_images': 'http://images.cocodataset.org/zips/test2015.zip',
        
        # Questions
        'train_questions': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip',
        'val_questions': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip',
        'test_questions': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip',
        
        # Annotations (not available for test set)
        'train_annotations': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip',
        'val_annotations': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip',
        
        # Additional files
        'test_dev_questions': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test-Dev_mscoco.zip',
        'complementary_pairs': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip'
    }
    
    def download_and_extract(url: str, target_dir: Path, desc: str) -> None:
        """Download and extract a zip file with progress bar."""
        try:
            zip_path = target_dir / f"temp_{Path(url).name}"
            
            # Download if not already exists
            if not zip_path.exists():
                print(f"\nDownloading {desc}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(zip_path, 'wb') as f, tqdm(
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    desc=f"Downloading {desc}"
                ) as pbar:
                    for data in response.iter_content(8192):
                        size = f.write(data)
                        pbar.update(size)
            
            # Extract
            print(f"Extracting {desc}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            # Clean up
            zip_path.unlink()
            print(f"✓ Successfully processed {desc}")
            
        except Exception as e:
            print(f"! Error processing {desc}: {str(e)}")
            if zip_path.exists():
                zip_path.unlink()
            raise
    
    try:
        '''
        # First try ParlAI method
        print("\n1. Attempting to set up using ParlAI...")
        try:
            opt = Opt({
                'task': 'vqa_v2',
                'datatype': 'train:ordered',
                'datapath': str(output_dir),
            })
            teacher = create_task_agent_from_taskname(opt)[0]
            print("✓ ParlAI setup successful")
            return
        except Exception as e:
            print(f"! ParlAI setup failed: {str(e)}")
            print("\n2. Falling back to direct downloads...")
        '''
        
        # Create directory structure
        images_dir = output_dir / 'images'
        questions_dir = output_dir / 'questions'
        annotations_dir = output_dir / 'annotations'
        
        for directory in [images_dir, questions_dir, annotations_dir]:
            directory.mkdir(exist_ok=True)
        
        # Group files by type for organized downloading
        file_groups = {
            'Images': [k for k in urls.keys() if 'images' in k],
            'Questions': [k for k in urls.keys() if 'questions' in k],
            'Annotations': [k for k in urls.keys() if 'annotations' in k]
        }
        
        # Download and extract all files by group
        for group_name, file_keys in file_groups.items():
            print(f"\nProcessing {group_name}...")
            for key in file_keys:
                url = urls[key]
                target_dir = (images_dir if 'images' in key else 
                            questions_dir if 'questions' in key else 
                            annotations_dir)
                download_and_extract(url, target_dir, key)
        
        print("\n3. Verifying download...")
        expected_dirs = {
            'train2014': images_dir / 'train2014',
            'val2014': images_dir / 'val2014',
            'test2015': images_dir / 'test2015'
        }
        
        expected_files = {
            'train_questions': questions_dir / 'v2_Questions_Train_mscoco.json',
            'val_questions': questions_dir / 'v2_Questions_Val_mscoco.json',
            'test_questions': questions_dir / 'v2_Questions_Test_mscoco.json',
            'train_annotations': annotations_dir / 'v2_Annotations_Train_mscoco.json',
            'val_annotations': annotations_dir / 'v2_Annotations_Val_mscoco.json'
        }
        
        # Verify directories
        print("\nVerifying image directories:")
        for name, path in expected_dirs.items():
            if path.exists() and any(path.iterdir()):
                print(f"✓ Found {name} directory with files")
            else:
                print(f"! Warning: {name} directory is missing or empty")
        
        # Verify files
        print("\nVerifying question and annotation files:")
        for name, path in expected_files.items():
            if path.exists():
                print(f"✓ Found {name} file")
            else:
                print(f"! Warning: {name} file is missing")
        
        print(f"\n✓ VQA-v2 dataset successfully set up in {output_dir}")
        
    except Exception as e:
        print(f"\n! Error during VQA-v2 setup: {str(e)}")
        raise

    # Verify final structure matches what's expected by load_benchmark_dataset
    missing_files = []
    for group_name, group_items in expected_structure.items():
        for item_name, path in group_items.items():
            if not path.exists() or (path.is_dir() and not any(path.iterdir())):
                missing_files.append(f"{group_name}/{item_name}")
    
    if missing_files:
        raise DatasetLoadError(
            f"VQA-v2 setup incomplete. Missing: {', '.join(missing_files)}"
        )
    
    return output_dir

def setup_mmmu(output_dir: Path) -> Dict[str, Dataset]:
    """Setup MMMU dataset by combining all subjects."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Setting up MMMU Dataset in {output_dir} ===")
    
    # Create images directory
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    # Define split mapping and file names
    splits_map = {
        'train': 'dev',
        'validation': 'validation',
        'test': 'test'
    }
    
    # Check if cached files exist
    cached_files = {
        split: output_dir / f"mmmu_{split}.pt"
        for split in splits_map.keys()
    }
    
    # If all cached files exist, load and return them
    if all(f.exists() for f in cached_files.values()):
        print("Found cached MMMU datasets, loading from disk...")
        combined_datasets = {}
        for split, file_path in cached_files.items():
            try:
                combined_datasets[split] = torch.load(file_path)
                print(f"✓ Loaded {split} split: {len(combined_datasets[split])} examples")
            except Exception as e:
                print(f"! Error loading {split} split: {str(e)}")
                break
        else:  # If no break occurred (all loads successful)
            n_examples = sum(len(dataset) for dataset in combined_datasets.values())
            print(f"\n✓ Loaded MMMU datasets from cache with {n_examples} total examples")
            return combined_datasets
    
    # If we get here, we need to download and process the datasets
    print("1. Downloading all MMMU subjects...")
    
    # List of all MMMU subjects
    subjects = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 
               'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 
               'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 
               'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 
               'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 
               'Physics', 'Psychology', 'Public_Health', 'Sociology']
    
    combined_datasets = {}
    
    for target_split, hf_split in splits_map.items():
        print(f"\nLoading {target_split} split (HF split: {hf_split}):")
        split_datasets = []
        
        split_dir = images_dir / target_split
        split_dir.mkdir(exist_ok=True)
        
        for subject in subjects:
            print(f"  - Downloading {subject}...")
            try:
                dataset = load_dataset(
                    "MMMU/MMMU",
                    subject,
                    split=hf_split,
                    cache_dir=str(output_dir)
                )
                
                # Process each example to save images to disk
                processed_examples = []
                for idx, item in enumerate(dataset):
                    try:
                        if 'image' in item and item['image']:
                            # Create unique filename for this image
                            image_filename = f"{subject}_{idx}.jpg"
                            image_path = split_dir / image_filename
                            
                            if not image_path.exists():  # Only save if not already exists
                                # Decode and save image
                                image_bytes = BytesIO(base64.b64decode(item['image']))
                                image = Image.open(image_bytes)
                                image.save(image_path)
                            
                            # Create standardized example format
                            processed_example = {
                                'question': item['question'],
                                'answer': item['answer'],
                                'image_path': str(image_path),
                                'split': target_split,
                                'subject': subject  # Keep subject as metadata
                            }
                            processed_examples.append(processed_example)
                            
                    except Exception as e:
                        print(f"    ! Error processing example {idx}: {str(e)}")
                        continue
                
                # Convert to Dataset
                if processed_examples:
                    split_datasets.append(Dataset.from_list(processed_examples))
                    print(f"    ✓ Processed {len(processed_examples)} examples")
                
            except Exception as e:
                print(f"    ! Error loading {subject}: {str(e)}")
                continue
        
        # Combine all subjects for this split
        if split_datasets:
            combined_datasets[target_split] = concatenate_datasets(split_datasets)
            print(f"  ✓ Combined {target_split} split: {len(combined_datasets[target_split])} total examples")
            
            # Save this split immediately
            try:
                torch.save(combined_datasets[target_split], cached_files[target_split])
                print(f"  ✓ Saved {target_split} split to {cached_files[target_split]}")
            except Exception as e:
                print(f"  ! Error saving {target_split} split: {str(e)}")
    
    print("\n2. Verifying combined datasets:")
    for split, dataset in combined_datasets.items():
        print(f"  - {split}: {len(dataset)} examples")
    
    n_examples = sum(len(dataset) for dataset in combined_datasets.values())
    print(f"\n✓ MMMU dataset successfully set up in {output_dir} with {n_examples} examples")
    
    return combined_datasets

def ensure_dataset(benchmark: str, data_dir: Optional[str] = None) -> Path:
    """Ensure dataset is downloaded and return its path."""
    print(f"\n=== Checking {benchmark.upper()} Dataset ===")
    
    if benchmark == 'chartqa':
        output_dir = Path(data_dir or CHARTQA_DIR)
        print(f"ChartQA directory: {output_dir.absolute()}")
        dataset_dir = output_dir / 'ChartQA Dataset'
        train_file = dataset_dir / 'train' / 'train_augmented.json'
        print(f"Looking for dataset at: {train_file.absolute()}")
        
        if not train_file.exists():
            print("! ChartQA dataset not found. Starting download...")
            download_chartqa(output_dir)
            print(f"Verifying download at: {dataset_dir.absolute()}")
            if train_file.exists():
                print(f"✓ Found train file: {train_file.absolute()}")
            else:
                print(f"! Warning: Expected train file not found at {train_file.absolute()}")
        else:
            print("✓ ChartQA dataset already exists")
            print(f"  - Train file: {train_file.absolute()}")
            print(f"  - Dataset dir: {dataset_dir.absolute()}")
        return dataset_dir
        
    elif benchmark == 'vqav2':
        output_dir = Path(VQA_V2_DIR)
        print(f"VQA-v2 directory: {output_dir.absolute()}")
        if not (output_dir / 'images').exists():
            print("! VQA-v2 dataset not found. Starting setup...")
            setup_vqa_v2(output_dir)
        else:
            print("✓ VQA-v2 dataset already exists")
            print(f"  - Images: {(output_dir / 'images').absolute()}")
            print(f"  - Questions: {(output_dir / 'questions').absolute()}")
            
    elif benchmark == 'mmmu':
        output_dir = Path(MMMU_DIR)
        print(f"MMMU directory: {output_dir.absolute()}")
        if not output_dir.exists():
            print("! MMMU dataset not found. Starting setup...")
            setup_mmmu(output_dir)
        else:
            print("✓ MMMU dataset already exists")
            print(f"  - Cache directory: {output_dir.absolute()}")
            
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
        
    return output_dir

def evaluate_chartqa_answer(predicted: str, ground_truth: str) -> float:
    """
    ChartQA official evaluation metric.
    
    Args:
        predicted: Model's predicted answer
        ground_truth: Ground truth answer
    
    Returns:
        Score between 0 and 1
        
    Raises:
        MetricCalculationError: If there's an error calculating the metric
    """
    try:
        predicted = predicted.strip().lower()
        ground_truth = ground_truth.strip().lower()
        
        try:
            # Try numerical comparison first
            pred_num = float(predicted.replace(',', ''))
            true_num = float(ground_truth.replace(',', ''))
            # Allow 5% tolerance for numerical answers (official metric)
            return float(abs(pred_num - true_num) / max(abs(true_num), 1e-6) <= 0.05)
        except ValueError:
            # If not numerical, do exact string matching
            return float(predicted == ground_truth)
    except Exception as e:
        raise MetricCalculationError(f"Error calculating ChartQA metric: {str(e)}")

def evaluate_vqav2_answer(predicted: str, ground_truth: List[str]) -> float:
    """
    VQAv2 official evaluation metric.
    
    Args:
        predicted: Model's predicted answer
        ground_truth: List of ground truth answers from different annotators
    
    Returns:
        Score between 0 and 1
        
    Raises:
        MetricCalculationError: If there's an error calculating the metric
    """
    try:
        predicted = predicted.strip().lower()
        answer_count: Dict[str, int] = {}
        
        for ans in ground_truth:
            ans = ans.strip().lower()
            answer_count[ans] = answer_count.get(ans, 0) + 1
        
        return min(answer_count.get(predicted, 0) / 3, 1)
    except Exception as e:
        raise MetricCalculationError(f"Error calculating VQAv2 metric: {str(e)}")

def evaluate_mmmu_answer(predicted: str, ground_truth: str, is_long_form: bool = False) -> float:
    """
    MMMU official evaluation metrics.
    
    Args:
        predicted: Model's predicted answer
        ground_truth: Ground truth answer
        is_long_form: Whether to use Rouge-L (True) or exact match (False)
    
    Returns:
        Score between 0 and 1
        
    Raises:
        MetricCalculationError: If there's an error calculating the metric
    """
    try:
        predicted = predicted.strip().lower()
        ground_truth = ground_truth.strip().lower()
        
        if is_long_form:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(ground_truth, predicted)
            return scores['rougeL'].fmeasure
        else:
            return float(predicted == ground_truth)
    except Exception as e:
        raise MetricCalculationError(f"Error calculating MMMU metric: {str(e)}")

def save_processed_dataset(dataset: Union[Dataset, List, Dict], benchmark: str, split: str, num_samples: Optional[int] = None) -> None:
    """Save processed dataset to disk cache."""
    try:
        # Extract dataset if it's in a dictionary
        if isinstance(dataset, dict) and 'dataset' in dataset:
            dataset = dataset['dataset']
            
        # Convert list to Dataset if needed
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
        
        # Truncate if needed
        if num_samples is not None:
            dataset = dataset.select(range(min(len(dataset), num_samples)))
            
        cache_key = f"{benchmark}_{split}_{num_samples}"
        cache_path = _CACHE_DIR / f"{cache_key}.pt"
        torch.save(dataset, cache_path)
        logger.info(f"Saved processed dataset to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save dataset cache: {str(e)}")

def load_processed_dataset(benchmark: str, split: str, num_samples: Optional[int] = None) -> Optional[Dataset]:
    """Load processed dataset from disk cache."""
    try:
        cache_key = f"{benchmark}_{split}_{num_samples}"
        cache_path = _CACHE_DIR / f"{cache_key}.pt"
        if cache_path.exists():
            logger.info(f"Loading cached dataset from {cache_path}")
            data = torch.load(cache_path)
            
            # Extract dataset if it's in a dictionary
            if isinstance(data, dict) and 'dataset' in data:
                dataset = data['dataset']
            else:
                dataset = data
            
            # Define features based on benchmark
            if benchmark == 'vqav2':
                features = Features({
                    'question': Value('string'),
                    'answer': Value('string'),
                    'answers': Sequence(Value('string')),
                    'image_path': Value('string'),
                    'split': Value('string'),
                    'question_id': Value('string'),
                    'image_id': Value('string'),
                    'data_type': Value('string'),
                    'task_type': Value('string')
                })
            else:
                # Default features for other benchmarks
                features = Features({
                    'question': Value('string'),
                    'answer': Value('string'),
                    'image_path': Value('string'),
                    'split': Value('string')
                })
            
            # Convert to Dataset with appropriate features
            if isinstance(dataset, list):
                # Convert list items to have consistent types
                processed_records = []
                for item in dataset:
                    if benchmark == 'vqav2':
                        record = {
                            'question': str(item.get('question', '')),
                            'answer': str(item.get('answer', '')),
                            'answers': [str(a) for a in item.get('answers', [])],
                            'image_path': str(item.get('image_path', '')),
                            'split': str(item.get('split', split)),
                            'question_id': str(item.get('question_id', '')),
                            'image_id': str(item.get('image_id', '')),
                            'data_type': str(item.get('data_type', '')),
                            'task_type': str(item.get('task_type', ''))
                        }
                    else:
                        record = {
                            'question': str(item.get('question', '')),
                            'answer': str(item.get('answer', '')),
                            'image_path': str(item.get('image_path', '')),
                            'split': str(item.get('split', split))
                        }
                    processed_records.append(record)
                
                dataset = Dataset.from_list(processed_records, features=features)
            
            elif isinstance(dataset, Dataset):
                # Verify and convert features if needed
                if set(dataset.features.keys()) != set(features.keys()):
                    # Convert Dataset to list and back to ensure correct features
                    processed_records = []
                    for item in dataset:
                        if benchmark == 'vqav2':
                            record = {
                                'question': str(item.get('question', '')),
                                'answer': str(item.get('answer', '')),
                                'answers': [str(a) for a in item.get('answers', [])],
                                'image_path': str(item.get('image_path', '')),
                                'split': str(item.get('split', split)),
                                'question_id': str(item.get('question_id', '')),
                                'image_id': str(item.get('image_id', '')),
                                'data_type': str(item.get('data_type', '')),
                                'task_type': str(item.get('task_type', ''))
                            }
                        else:
                            record = {
                                'question': str(item.get('question', '')),
                                'answer': str(item.get('answer', '')),
                                'image_path': str(item.get('image_path', '')),
                                'split': str(item.get('split', split))
                            }
                        processed_records.append(record)
                    dataset = Dataset.from_list(processed_records, features=features)
            
            return dataset
                
    except Exception as e:
        logger.warning(f"Failed to load dataset cache: {str(e)}")
    return None

def preload_images(dataset_dict):
    """
    Preload images from dataset dictionary.
    
    Args:
        dataset_dict: Dictionary containing 'dataset' key with the actual dataset
    """
    images = {}
    
    # Extract dataset from dictionary if needed
    if isinstance(dataset_dict, dict) and 'dataset' in dataset_dict:
        dataset = dataset_dict['dataset']
    else:
        dataset = dataset_dict
    
    # Skip if dataset is empty
    if not dataset:
        return {'images': images}
        
    # Convert Dataset to list if needed
    if isinstance(dataset, Dataset):
        dataset = list(dataset)
    elif not isinstance(dataset, list):
        logger.warning(f"Unexpected dataset type: {type(dataset)}")
        return {'images': images}
        
    if not dataset:  # Check again after conversion
        return {'images': images}
        
    # Get first item to determine structure
    first_item = dataset[0]
    
    # Handle different dataset structures
    if isinstance(first_item, dict):
        image_paths = [item['image_path'] for item in dataset if 'image_path' in item]
    elif isinstance(first_item, str):
        # If items are strings, assume they are direct image paths
        image_paths = dataset
    else:
        logger.warning(f"Unexpected dataset item type: {type(first_item)}")
        return {'images': images}

    # Load images
    for path in tqdm(image_paths, desc="Loading images"):
        try:
            if isinstance(path, str):
                images[path] = Image.open(path).convert('RGB')
            elif isinstance(path, list):
                # Handle multiple images per item (e.g., ChartQA)
                for p in path:
                    images[p] = Image.open(p).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {str(e)}")
    
    return {'images': images}

from jinja2 import Template

def get_chat_template(suffix=""):
    # NOTE: for now, use the VICUNA_CHAT_TEMPLATE used by llava_hf
    # TODO: Eventually will want to ensure this works for other models
    template = """{% for message in messages %}
    {% if loop.index0 == 0 %}
    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    USER: {{ message['content'] }}
    {% elif message['role'] == 'user' %}
    USER: {{ message['content'] }}
    {% else %}
    ASSISTANT: {{ message['content'] }}{{ eos_token }}
    {% endif %}
    {% endfor %}
    {% if add_generation_prompt %}
    ASSISTANT: {{ suffix }}
    {% endif %}
    """
    suffix = suffix.replace("'","\\'")
    template = template.replace("{{ suffix }}", suffix)
    return template


def load_benchmark_dataset(
    benchmark: str,
    split: str = "validation",
    num_samples: Optional[int] = None,
    data_dir: Optional[str] = None,
    use_cache: Optional[bool] = None,
    preload_imgs: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Load a benchmark dataset with caching and optional image preloading.
    
    Returns:
        Dict with keys:
            - dataset: List[Dict[str, str]]
            - images: Optional[Dict[str, Image.Image]]
    """
    # Use environment variables if not explicitly overridden
    use_cache = _USE_CACHE if use_cache is None else use_cache
    preload_imgs = _PRELOAD_IMAGES if preload_imgs is None else preload_imgs
    
    cache_key = f"{benchmark}_{split}_{num_samples}"
    
    if use_cache:
        # Check memory cache
        if cache_key in _DATASET_CACHE:
            logger.info(f"Using in-memory cached dataset for {cache_key}")
            if len(_DATASET_CACHE[cache_key]["dataset"]) == 0:
                print(f"Warning: In-memory cached dataset for {cache_key} is empty")
                del _DATASET_CACHE[cache_key]
            else:
                print(f"In-memory cached dataset for {cache_key} has {len(_DATASET_CACHE[cache_key]['dataset'])} samples")
                return _DATASET_CACHE[cache_key]
            
        # Check disk cache
        cached_dataset = load_processed_dataset(benchmark, split, num_samples)
        if cached_dataset is not None:
            result = {'dataset': cached_dataset}
            if preload_imgs:
                result.update(preload_images(result))
            if len(result['dataset']) > 0:
                _DATASET_CACHE[cache_key] = result
                print(f"Loaded cached dataset for {cache_key} with {len(result['dataset'])} samples")
                return result
    
    # Load fresh dataset
    logger.info(f"Loading fresh dataset for {benchmark}")
    dataset_dir = ensure_dataset(benchmark, data_dir)
    
    if benchmark == 'chartqa':
        dataset = get_chartqa_dataset(split, "local", str(dataset_dir))
        # Keep as Dataset, no conversion needed since get_chartqa_dataset now returns correct format
        
    elif benchmark == 'vqav2':
        # Ensure dataset is downloaded and get its path
        dataset_dir = ensure_dataset(benchmark, data_dir)
        
        # Define paths based on setup_vqa_v2's structure
        images_dir = dataset_dir / 'images'
        questions_dir = dataset_dir / 'questions'
        annotations_dir = dataset_dir / 'annotations'
        
        # Map split names to file patterns
        split_map = {
            'train': ('train2014', 'v2_Questions_Train_mscoco.json', 'v2_Annotations_Train_mscoco.json'),
            'validation': ('val2014', 'v2_Questions_Val_mscoco.json', 'v2_Annotations_Val_mscoco.json'),
            'test': ('test2015', 'v2_Questions_Test_mscoco.json', None)  # No annotations for test
        }
        
        if split not in split_map:
            raise ValueError(f"Invalid split '{split}' for VQA-v2. Must be one of {list(split_map.keys())}")
        
        img_dir_name, question_file, annotation_file = split_map[split]
        
        # Load questions with proper structure handling
        question_path = questions_dir / question_file
        try:
            with open(question_path) as f:
                questions_data = json.load(f)
                
                # Validate question file structure
                required_keys = {'questions', 'info', 'task_type', 'data_type', 'data_subtype', 'license'}
                if not all(key in questions_data for key in required_keys):
                    missing_keys = required_keys - set(questions_data.keys())
                    raise ValueError(f"Question file missing required keys: {missing_keys}")
                
                # Log dataset info
                print(f"\nVQA-v2 Dataset Info:")
                print(f"  Task Type: {questions_data['task_type']}")
                print(f"  Data Type: {questions_data['data_type']}")
                print(f"  Version: {questions_data['info']['version']}")
                print(f"  Split: {split}")
                
                # Extract questions into a lookup dictionary
                questions = {
                    q['question_id']: {
                        'question_id': q['question_id'],
                        'image_id': q['image_id'],
                        'question': q['question']
                    }
                    for q in questions_data['questions']
                }
                
        except json.JSONDecodeError as e:
            raise DatasetLoadError(f"Failed to parse questions file {question_file}: {str(e)}")
        except KeyError as e:
            raise DatasetLoadError(f"Malformed question data in {question_file}: {str(e)}")
        
        # Load annotations if available
        answers = {}
        if annotation_file:
            annotation_path = annotations_dir / annotation_file
            try:
                with open(annotation_path) as f:
                    annotations_data = json.load(f)
                    for ann in annotations_data['annotations']:
                        answers[ann['question_id']] = [a['answer'] for a in ann['answers']]
            except Exception as e:
                raise DatasetLoadError(f"Failed to load annotations from {annotation_file}: {str(e)}")
        
        # Create dataset with consistent types
        dataset = []
        image_dir = images_dir / img_dir_name
        
        for question_id, question_data in questions.items():
            try:
                record = {
                    'question': str(question_data['question']),
                    'question_id': str(question_data['question_id']),  # Convert to string for consistency
                    'image_id': str(question_data['image_id']),  # Convert to string for consistency
                    'image_path': str(image_dir / f"COCO_{img_dir_name}_{question_data['image_id']:012d}.jpg"),
                    'split': str(split),
                    'data_type': str(questions_data['data_type']),
                    'task_type': str(questions_data['task_type'])
                }
                
                # Add answers if available (not for test set)
                if question_id in answers:
                    record['answers'] = [str(a) for a in answers[question_id]]  # Convert all answers to strings
                    record['answer'] = str(answers[question_id][0])  # Use first answer as primary
                else:
                    record['answers'] = []
                    record['answer'] = ""  # Empty for test set
                
                dataset.append(record)
                
            except Exception as e:
                print(f"! Warning: Failed to process question {question_id}: {str(e)}")
                continue
        
        # Truncate if needed
        if num_samples is not None:
            dataset = dataset[:num_samples]
        
        print(f"\nLoaded VQA-v2 {split} split with {len(dataset)} samples")
        if len(dataset) == 0:
            print(f"! Warning: No samples found in {split} split")
            print(f"  Questions file: {question_path}")
            print(f"  Images directory: {image_dir}")
            if annotation_file:
                print(f"  Annotations file: {annotation_path}")
        
        result = {'dataset': dataset}
        
        # Preload images if requested
        if preload_imgs:
            result.update(preload_images(result))
            print(f"Preloaded {len(result.get('images', {}))} images")
        
        # Cache if enabled
        if use_cache and len(dataset) > 0:
            try:
                save_processed_dataset(dataset, benchmark, split, num_samples)
                print(f"Saved processed dataset to cache")
            except Exception as e:
                print(f"! Warning: Failed to save to cache: {str(e)}")
        
        return result

    elif benchmark == 'mmmu':
        mmmu_split = {
            'train': 'dev',
            'validation': 'validation',
            'test': 'test'
        }.get(split, split)
        
        combined_dataset = setup_mmmu(dataset_dir)
        dataset = combined_dataset[mmmu_split]
        
        # Convert base64 images to PIL Images
        processed_dataset = []
        for item in dataset:
            if 'image' in item and item['image']:
                try:
                    # Decode base64 image
                    image_bytes = BytesIO(base64.b64decode(item['image']))
                    image = Image.open(image_bytes)
                    processed_item = {
                        'question': item['question'],
                        'answer': item['answer'],
                        'image': image  # Store the PIL Image directly
                    }
                    processed_dataset.append(processed_item)
                except Exception as e:
                    logger.warning(f"Failed to process image: {str(e)}")
                    print(f"Failed to process image: {str(e)}")
                    continue
        dataset = processed_dataset
    
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    print(f"Loaded dataset for {benchmark} with {len(dataset)} samples")
    
    # Truncate if needed
    if num_samples is not None and isinstance(dataset, list):
        dataset = dataset[:num_samples]
    
    result = {'dataset': dataset}
    
    if preload_imgs:
        result.update(preload_images(result))
    
    _DATASET_CACHE[cache_key] = result
    
    if use_cache:
        save_processed_dataset(dataset, benchmark, split, num_samples)
    
    return result

def evaluate_model(
    model: Any,
    processor: Any,
    benchmark: str,
    split: str = "validation",
    num_samples: Optional[int] = None,
    prefix: str = ""
) -> Dict[str, float]:
    """Evaluate model using lmms-eval framework"""
    try:
        # Define allowed splits for each benchmark based on YAML configs
        allowed_splits = {
            'chartqa': ['test'],  # From chartqa.yaml: test_split: test
            'vqav2': ['validation', 'test'],  # From vqav2_val.yaml and vqav2_test.yaml
            'mmmu': ['validation', 'test']  # From mmmu_val.yaml and mmmu_test.yaml
        }
        
        if benchmark not in allowed_splits:
            raise ValueError(
                f"Unsupported benchmark: {benchmark}. "
                f"Supported benchmarks: {list(allowed_splits.keys())}"
            )
            
        if split not in allowed_splits[benchmark]:
            raise ValueError(
                f"Split '{split}' not available for benchmark '{benchmark}'. "
                f"Available splits: {allowed_splits[benchmark]}"
            )
        
        # Map benchmark names to lmms-eval task names based on YAML configs
        task_map = {
            'chartqa': 'chartqa',  # From chartqa.yaml
            'vqav2': 'vqav2_val' if split == "validation" else "vqav2_test",  # From vqav2_val.yaml/vqav2_test.yaml
            'mmmu': 'mmmu_val' if split == "validation" else "mmmu_test"  # From mmmu_val.yaml/mmmu_test.yaml
        }
            
        task_name = task_map[benchmark]
        task_dict = get_task_dict([task_name])
        #print(f"Task dict is {task_dict}")

        # Initialize task manager to load tasks
        task_manager = lmms_eval.tasks.TaskManager()
       
        # Configure task parameters
        task_config = {
           "num_fewshot": 0,  # Can be parameterized if needed
           "num_samples": num_samples,
           "split": split,
           #"prefix": prefix
        }
        # Get task dict with configuration
        task_dict = lmms_eval.tasks.get_task_dict(
           [task_name],
           task_manager,
        )
        #print(f"Task dict is {task_dict}")
        #task_dict[task_name].suffix = prefix
        try:
            model.chat_template = get_chat_template(prefix)
        except Exception as e:
            print(e)
        try:
            model.suffix = prefix
        except Exception as e:
            print(e)
        print(f"prefix: {prefix}")
        #print(f"Model chat_template: {model.chat_template}")

        eval_config = {
            "model": model,
            "tasks": [task_name],
            "batch_size": 1,
            "num_fewshot": 0,
            "limit": num_samples,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            #"use_cache": f"{CACHE_DIR}/{task_name}",
            #"cache_requests": True,
            "log_samples": True,
        }
        #print(f"Eval config is {eval_config}")
        
        # Run evaluation
        try:
            model_attr = model._model
        except Exception as e:
            print(e)
        results = simple_evaluate(**eval_config)
        try:
            model._model = model_attr
        except Exception as e:
            print(e)
            
        # Configure evaluation
        '''
        eval_config = {
            "lm": model,
            "task_dict": task_dict,
            "limit": num_samples,
            #"use_cache": f"{CACHE_DIR}/{task_name}",
            "cache_requests": True,
            "log_samples": True
        }
        print(f"eval_config: {eval_config}")
        
        # Run evaluation
        results = evaluate(**eval_config)
        '''
        #print(results)
        task_results = results['results'][task_name]
        print(task_results)
        
        # Extract metrics based on benchmark and YAML metric_list configurations
        metrics = {}
        if benchmark == 'chartqa':
            # From chartqa.yaml
            metrics['relaxed_overall'] = task_results['relaxed_overall,none']
            metrics['relaxed_human_split'] = task_results['relaxed_human_split,none']
            #metrics['relaxed_augmented_split'] = task_results['relaxed_augmented_split,none']
            
        elif benchmark == 'vqav2':
            if split == "validation":
                # From vqav2_val.yaml
                metrics['exact_match'] = task_results['exact_match,none']
            else:
                # From vqav2_test.yaml
                metrics['submission'] = task_results['submission,none']
                
        elif benchmark == 'mmmu':
            if split == "validation":
                # From mmmu_val.yaml
                metrics['mmmu_acc'] = task_results['mmmu_acc,none']
            else:
                # From mmmu_test.yaml
                metrics['submission'] = task_results['submission,none']
                
        return metrics

    except Exception as e:
        raise EvaluationError(f"Error during evaluation: {str(e)}")

def save_results(benchmark: str, results: List[Dict], metrics: Dict[str, float]) -> None:
    """
    Save evaluation results to a JSON file.
    
    Args:
        benchmark: Name of the benchmark
        results: List of individual evaluation results
        metrics: Dictionary of computed metrics
        
    Raises:
        IOError: If there's an error saving the results
    """
    try:
        output = {
            'benchmark': benchmark,
            'metrics': metrics,
            'results': results
        }
        
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / f'eval_results_{benchmark}.json', 'w') as f:
            json.dump(output, f, indent=2)
    except Exception as e:
        raise IOError(f"Error saving results: {str(e)}")

def load_results(benchmark: str) -> Dict:
    """
    Load previously saved evaluation results.
    
    Args:
        benchmark: Name of the benchmark
        
    Returns:
        Dictionary containing evaluation results
        
    Raises:
        IOError: If there's an error loading the results
    """
    try:
        with open(Path('results') / f'eval_results_{benchmark}.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        raise IOError(f"Error loading results: {str(e)}") 

def resize_image_aspect_ratio(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Resize image to fit within target dimensions while maintaining aspect ratio."""
    aspect_ratio = img.width / img.height
    if aspect_ratio > target_width / target_height:
        # Width is the limiting factor
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Height is the limiting factor
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS) 

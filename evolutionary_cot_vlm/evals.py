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
from constants import CHARTQA_DIR, VQA_V2_DIR, MMMU_DIR
import zipfile
import pickle
from functools import lru_cache
import base64
from transformers import LlavaProcessor, Blip2Processor

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
    dataset_dict: Dict[str, Any],
    benchmark: str,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Evaluate model on a pre-loaded dataset.
    
    Args:
        model: The model to evaluate
        processor: The model's processor/tokenizer
        dataset_dict: Dictionary containing dataset and optionally preloaded images
        benchmark: Name of the benchmark
        prefix: Prefix to add to questions
        
    Returns:
        Evaluation metrics
    """
    try:
        dataset = dataset_dict['dataset']
        preloaded_images = dataset_dict.get('images', {})
        
        print(f"\n=== Starting Evaluation on {benchmark.upper()} ===")
        print(f"Using provided dataset with {len(dataset)} samples")
        print(f"Preloaded images: {len(preloaded_images)}")
        
        # Initialize metrics
        correct = 0
        total = 0
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        # Initialize counters
        processed_count = 0
        skipped_count = 0

        results: List[Dict[str, Any]] = []
        
        # If using BLIP-2, set required attributes
        if isinstance(processor, Blip2Processor):
            processor.image_processor.is_vqa = True
            processor.image_processor.patch_size = 14
            processor.tokenizer.truncation_side = "left"
        # If using LLaVa, set required attributes
        elif isinstance(processor, LlavaProcessor):
            # Set these attributes before any processing
            processor.image_processor.size = {"height": 336, "width": 336}  # LLaVa's default size
            processor.image_processor.patch_size = 14  # Set on image_processor instead
            processor.image_processor.vision_feature_select_strategy = "full"  # Set on image_processor instead
            processor.patch_size = 14  # Standard patch size for LLaVa
            processor.vision_feature_select_strategy = "full"  # Use full feature strategy
            processor.is_vqa = True  # Add VQA flag for proper processing
            logger.info("LLaVa processor configured with:")
            logger.info(f"  - Image size: {processor.image_processor.size}")
            logger.info(f"  - Patch size: {processor.image_processor.patch_size}")
            logger.info(f"  - Feature strategy: {processor.image_processor.vision_feature_select_strategy}")

        with torch.no_grad():
            for idx, item in enumerate(tqdm(dataset)):
                try:
                    if benchmark == 'chartqa' and isinstance(processor, LlavaProcessor) and isinstance(item.get('image_path'), list):
                                images = []
                                for img_path in item.get('image_path'):
                                    if img_path in preloaded_images:
                                        images.append(preloaded_images[img_path])
                                    else:
                                        images.append(Image.open(img_path))
                                
                                # Calculate grid dimensions (2x3 grid for 5 images)
                                n_images = len(images)
                                grid_size = (2, 3)  # rows x cols
                                
                                # Use LLaVa's expected image size for the final image
                                final_width = 336
                                final_height = 336
                                
                                # Calculate cell size to fit within final dimensions
                                cell_width = final_width // grid_size[1]
                                cell_height = final_height // grid_size[0]
                                
                                # Create new image with grid layout
                                combined_image = Image.new('RGB', (final_width, final_height))
                                
                                # Paste images into grid
                                for i, img in enumerate(images):
                                    if i >= grid_size[0] * grid_size[1]:
                                        break
                                    # Resize image to fit cell while maintaining aspect ratio
                                    img = resize_image_aspect_ratio(img, cell_width, cell_height)
                                    row = i // grid_size[1]
                                    col = i % grid_size[1]
                                    # Center image in its cell
                                    x_offset = col * cell_width + (cell_width - img.width) // 2
                                    y_offset = row * cell_height + (cell_height - img.height) // 2
                                    combined_image.paste(img, (x_offset, y_offset))
                                
                                image = combined_image
                    else:
                            # Original processing for non-ChartQA datasets
                            image_path = item.get('image_path')
                            if image_path in preloaded_images:
                                image = preloaded_images[image_path]
                            else:
                                image = Image.open(image_path)

                    # Process inputs
                    if isinstance(processor, Blip2Processor):
                        inputs = processor(
                            images=image,
                            text=item['question'] + prefix,
                            return_tensors="pt"
                        ).to(device)
                        
                    elif isinstance(processor, LlavaProcessor):
                        inputs = processor(
                            images=image,
                            text=item['question'] + prefix,
                            return_tensors="pt"
                        ).to(device)
                    else:
                        # Default processing for other models
                        inputs = processor(
                            images=image,
                            text=item['question'] + prefix,
                            return_tensors="pt"
                        ).to(device)

                    if benchmark == 'chartqa':
                        question = item['question'] + prefix
                        ground_truth = str(item['answer'])
                        is_long_form = False
                        
                    elif benchmark == 'vqav2':
                        question = item['question'] + prefix
                        ground_truth = item['answers']
                        is_long_form = False
                        
                    elif benchmark == 'mmmu':
                        if 'image' not in item:
                            skipped_count += 1
                            continue
                        processed_count += 1
                        image = item['image']  # Already a PIL Image
                        question = item['question'] + prefix
                        ground_truth = str(item['answer'])
                        is_long_form = len(ground_truth.split()) > 15 or '\n' in ground_truth

                    # Generate prediction
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50 if benchmark != 'mmmu' else 100,
                        num_beams=5,
                        early_stopping=True
                    )
                    
                    predicted = processor.decode(outputs[0], skip_special_tokens=True).lower()
                    
                    # Print first 3 items' predictions and ground truth
                    if idx < 3:
                        print(f"\n=== Item {idx + 1} Debug ===")
                        print(f"Question + Prefix: {item['question'] + prefix}")
                        print(f"Predicted: {predicted}")
                        print(f"Ground Truth: {ground_truth}")
                        print("========================")
                    
                    # Calculate score
                    if benchmark == 'chartqa':
                        score = evaluate_chartqa_answer(predicted, ground_truth)
                    elif benchmark == 'vqav2':
                        score = evaluate_vqav2_answer(predicted, ground_truth)
                    else:  # mmmu
                        score = evaluate_mmmu_answer(predicted, ground_truth, is_long_form)
                    
                    correct += score
                    total += 1
                    
                    results.append({
                        'question': question,
                        'ground_truth': ground_truth,
                        'predicted': predicted,
                        'score': float(score),
                        'is_long_form': is_long_form if benchmark == 'mmmu' else None
                    })

                except Exception as e:
                    logger.error(f"Error processing item: {str(e)}")
                    continue

        # Calculate metrics
        metrics: Dict[str, float] = {
            'accuracy': correct / total if total > 0 else 0.0,
            'total_samples': float(total),
            'correct_samples': float(correct)
        }
        
        # Add benchmark-specific metrics
        if benchmark == 'mmmu':
            long_form_results = [r for r in results if r.get('is_long_form', False)]
            short_form_results = [r for r in results if not r.get('is_long_form', False)]
            
            if long_form_results:
                metrics['long_form_rouge_l'] = sum(r['score'] for r in long_form_results) / len(long_form_results)
                metrics['long_form_count'] = float(len(long_form_results))
            
            if short_form_results:
                metrics['short_form_accuracy'] = sum(r['score'] for r in short_form_results) / len(short_form_results)
                metrics['short_form_count'] = float(len(short_form_results))

            print(f"\nMMU Stats: {processed_count} items processed, {skipped_count} items skipped (no images)")
        return metrics

    except Exception as e:
        print(f"\n! Error during evaluation: {str(e)}")
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

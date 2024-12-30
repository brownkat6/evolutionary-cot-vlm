from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import requests
from datasets import Dataset
import logging
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)

def download_file(url: str, output_path: Path, desc: str = None) -> None:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save file
        desc: Description for progress bar
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(output_path, 'wb') as f, tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        desc=desc
    ) as pbar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            pbar.update(size)

def download_chartqa_dataset(output_dir: str = "data/chartqa") -> str:
    """
    Download ChartQA dataset from official source.
    
    Args:
        output_dir: Directory to save dataset
        
    Returns:
        Path to dataset directory
    """
    # Base URL from verified repository
    base_url = "https://github.com/vis-nlp/ChartQA/raw/main/ChartQA%20Dataset"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Verified directory structure
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = output_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Download JSON file
        json_url = f"{base_url}/{split}/{split}_augmented.json"
        json_path = split_dir / f"{split}_augmented.json"
        
        if not json_path.exists():
            logger.info(f"Downloading {split}_augmented.json...")
            try:
                download_file(json_url, json_path, f"Downloading {split}_augmented.json")
            except Exception as e:
                raise RuntimeError(f"Failed to download {split}_augmented.json: {e}")
        
        # Create png directory
        png_dir = split_dir / 'png'
        png_dir.mkdir(exist_ok=True)
        
        # Load JSON to get image filenames
        try:
            with open(json_path) as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"Expected list in {split}_augmented.json, got {type(data)}")
                
                # Download images
                for item in tqdm(data, desc=f"Downloading {split} images"):
                    image_filename = item['imgname']
                    image_url = f"{base_url}/{split}/png/{image_filename}"
                    image_path = png_dir / image_filename
                    
                    if not image_path.exists():
                        try:
                            download_file(image_url, image_path, None)
                        except Exception as e:
                            logger.warning(f"Failed to download image {image_filename}: {e}")
                            continue
                
                logger.info(f"Verified {split}_augmented.json structure and downloaded images")
        except Exception as e:
            raise RuntimeError(f"Failed to process {split}_augmented.json: {e}")
    
    return str(output_path)

def load_local_chartqa(data_dir: str, split: str) -> Dataset:
    """
    Load ChartQA dataset from local files.
    
    Args:
        data_dir: Path to dataset directory
        split: One of ['train', 'validation', 'test']
        
    Returns:
        Dataset object
    """
    # Convert validation to val for file paths
    file_split = 'val' if split == 'validation' else split
    
    data_path = Path(data_dir) / file_split / f"{file_split}_augmented.json"
    png_dir = Path(data_dir) / file_split / 'png'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    if not png_dir.exists():
        raise FileNotFoundError(f"PNG directory not found: {png_dir}")
    
    with open(data_path) as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Invalid data format in {data_path}")
    
    # Verify required fields with correct names
    required_fields = {'imgname', 'query', 'label'}
    for item in data:
        missing_fields = required_fields - set(item.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
    
    return Dataset.from_dict({
        'question': [item['query'] for item in data],
        'answer': [str(item['label']) for item in data],
        'image_path': [str(png_dir / item['imgname']) for item in data],
        'split': [split] * len(data)
    })

def get_chartqa_dataset(split: str, method: str = "local", data_dir: Optional[str] = None) -> Dataset:
    """
    Get ChartQA dataset using specified method.
    
    Args:
        split: One of ['train', 'validation', 'test']
        method: One of ['local', 'download']
        data_dir: Optional path to local dataset
        
    Returns:
        Dataset object with fields:
            - question: str (question text)
            - answer: str (answer text)
            - image_path: str (full path to image file)
            - split: str (dataset split)
    """
    if split not in ['train', 'validation', 'test']:
        raise ValueError(f"Invalid split: {split}")
    
    if method == "local":
        if not data_dir:
            raise ValueError("data_dir must be provided for local loading")
        return load_local_chartqa(data_dir, split)
    
    elif method == "download":
        data_dir = download_chartqa_dataset()
        return load_local_chartqa(data_dir, split)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'local' or 'download'") 
from typing import Tuple, Any, Optional, List, Dict
import os
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM, 
    BlipProcessor, 
    BlipForConditionalGeneration,
    LlavaProcessor,
    LlavaForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizer
)
import torch
from evolutionary_cot_vlm.constants import CACHE_DIR
from PIL import Image

class ModelLoadError(Exception):
    """Custom exception for model loading errors."""
    pass

def load_model(model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a multimodal model and its processor/tokenizer.
    
    Args:
        model_name: One of ['blip2', 'llava', 'minigpt4', 'otter', 'molmo']
    
    Returns:
        Tuple of (model, processor/tokenizer)
    
    Raises:
        ValueError: If model_name is not supported
        ModelLoadError: If there's an error loading the model
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_map: dict[str, str] = {
            #"blip2": "Salesforce/blip2-opt-2.7b", # non-instruct-finetuned version
            "blip2": "Salesforce/blip2-opt-2.7b-coco", # coco-finetuned instructed version
            "llava": "llava-hf/llava-1.5-7b-hf",
            "minigpt4": "microsoft/minigpt4-7b",
            "otter": "luodian/otter-9b-hf",
            "molmo": "allenai/Molmo-7B-D-0924"
        }
        
        if model_name.lower() not in model_map:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(model_map.keys())}")
        
        model_path = model_map[model_name.lower()]
        
        if model_name.lower() == "blip2":
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            processor = Blip2Processor.from_pretrained(
                model_path,
                cache_dir=CACHE_DIR,
                trust_remote_code=True
            )
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_path,
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(device)
            
        elif model_name.lower() == "llava":
            processor = LlavaProcessor.from_pretrained(
                model_path,
                cache_dir=CACHE_DIR,
                trust_remote_code=True
            )
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(device)
            processor.patch_size = 14  # Standard patch size for LLaVa
            processor.vision_feature_select_strategy = "full"  # Use full feature strategy
            
        else:  # For minigpt4, otter, and molmo
            processor = AutoProcessor.from_pretrained(
                model_path,
                cache_dir=CACHE_DIR,
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(device)
        
        return model, processor
        
    except Exception as e:
        raise ModelLoadError(f"Error loading {model_name}: {str(e)}") 

from typing import Tuple, Any, Optional
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
from utils import CACHE_DIR

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
            "blip2": "Salesforce/blip2-opt-2.7b",
            "llava": "llava-hf/llava-1.5-7b-hf",
            "minigpt4": "microsoft/minigpt4-7b",
            "otter": "luodian/otter-9b-hf",
            "molmo": "ContextualAI/molmo-7b"
        }
        
        if model_name.lower() not in model_map:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(model_map.keys())}")
        
        model_path = model_map[model_name.lower()]
        
        if model_name.lower() == "blip2":
            processor = BlipProcessor.from_pretrained(
                model_path,
                cache_dir=CACHE_DIR
            )
            model = BlipForConditionalGeneration.from_pretrained(
                model_path,
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
        elif model_name.lower() == "llava":
            processor = LlavaProcessor.from_pretrained(
                model_path,
                cache_dir=CACHE_DIR
            )
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
        else:  # For minigpt4, otter, and molmo
            processor = AutoProcessor.from_pretrained(
                model_path,
                cache_dir=CACHE_DIR
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
        
        return model, processor
        
    except Exception as e:
        raise ModelLoadError(f"Error loading {model_name}: {str(e)}") 
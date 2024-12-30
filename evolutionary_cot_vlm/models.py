from typing import Tuple, Any, Optional
import torch
from evolutionary_cot_vlm.constants import CACHE_DIR
from lmms_eval.models.llava_hf import LlavaHf
from lmms_eval.models.blip2_hf import Blip2Hf
from lmms_eval.models.minigpt4_hf import MiniGPT4Hf
from lmms_eval.models.otter_hf import OtterHf

class ModelLoadError(Exception):
    """Custom exception for model loading errors."""
    pass

def load_model(model_name: str) -> Tuple[Any, Optional[Any]]:
    """
    Load a multimodal model and its processor/tokenizer.
    
    Args:
        model_name: One of ['blip2', 'llava', 'minigpt4', 'otter', 'molmo']
    
    Returns:
        Tuple of (model, processor). For lmms-eval native models, processor is None.
    """
    try:
        model_map: dict[str, str] = {
            "blip2": "Salesforce/blip2-opt-2.7b-coco",
            "llava": "llava-hf/llava-1.5-7b-hf",
            "minigpt4": "microsoft/minigpt4-7b",
            "otter": "luodian/otter-9b-hf",
            "molmo": "allenai/Molmo-7B-D-0924"
        }
        
        if model_name.lower() not in model_map:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(model_map.keys())}")
        
        model_path = model_map[model_name.lower()]
        
        # Use native lmms-eval implementations where available
        if model_name.lower() == "llava":
            model = LlavaHf(
                pretrained=model_path,
                batch_size=1,
                trust_remote_code=True
            )
            return model, None
            
        elif model_name.lower() == "blip2":
            model = Blip2Hf(
                pretrained=model_path,
                batch_size=1,
                trust_remote_code=True
            )
            return model, None
            
        elif model_name.lower() == "minigpt4":
            model = MiniGPT4Hf(
                pretrained=model_path,
                batch_size=1,
                trust_remote_code=True
            )
            return model, None
            
        elif model_name.lower() == "otter":
            model = OtterHf(
                pretrained=model_path,
                batch_size=1,
                trust_remote_code=True
            )
            return model, None
            
        else:  # Fallback for Molmo using custom wrapper
            device = "cuda" if torch.cuda.is_available() else "cpu"
            from transformers import AutoProcessor, AutoModelForCausalLM
            
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

from typing import Tuple, Any, Optional
import torch
from evolutionary_cot_vlm.constants import CACHE_DIR
from lmms_eval.models.llava_hf import LlavaHf
#from lmms_eval.models.blip2_hf import Blip2Hf
from lmms_eval.models.instructblip import InstructBLIP
from lmms_eval.models.batch_gpt4 import BatchGPT4
from lmms_eval.models.claude import Claude
from lmms_eval.models.fuyu import Fuyu
from lmms_eval.models.mantis import Mantis
#from lmms_eval.models.minigpt4_hf import MiniGPT4Hf
#from lmms_eval.models.otter_hf import OtterHf
# TODO: fix the gpt4, otter_hf, and molmo model loading
from lmms_eval.evaluator import evaluate
from types import SimpleNamespace

# NOTE: required environment variables:
# ANTHROPIC_API_KEY
# OPENAI_API_URL

# TODO: fix lmms-eval's batch_gpt4 model loading to import List etc. 

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
            #"blip2": "Salesforce/blip2-opt-2.7b-coco",
            "blip2": "Salesforce/instructblip-vicuna-7b",
            "llava": "llava-hf/llava-1.5-7b-hf",
            #"minigpt4": "microsoft/minigpt4-7b",
            "minigpt4": "gpt-4o",
            #"otter": "luodian/otter-9b-hf",
            "claude": "claude-3-opus-20240229",
            "molmo": "allenai/Molmo-7B-D-0924",
            "mantis": "TIGER-Lab/Mantis-8B-clip-llama3",
            "fuyu": "adept/fuyu-8b",
        }
        
        if model_name.lower() not in model_map:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(model_map.keys())}")
        
        model_path = model_map[model_name.lower()]
        
        # Use native lmms-eval implementations where available
        if model_name.lower() == "llava":
            model = LlavaHf(
                pretrained=model_path,
                batch_size=1,
                trust_remote_code=True,
                device_map="auto",
                cache_dir=CACHE_DIR,
            )
            return model, None
            
        elif model_name.lower() == "blip2":
            model = InstructBLIP(
                pretrained=model_path,
                batch_size=1,
                cache_dir=CACHE_DIR,
                #trust_remote_code=True,
                #device_map="auto",
            )
            return model, None
        elif model_name.lower() == "mantis":
            model = Mantis(
                pretrained=model_path,
                batch_size=1,
                #trust_remote_code=True,
                device_map="auto",
                cache_dir=CACHE_DIR,
            )
            return model, None
        if model_name.lower() == "fuyu":
            model = Fuyu(
                pretrained=model_path,
                batch_size=1,
                #trust_remote_code=True,
                #device_map="auto",
                cache_dir=CACHE_DIR,
            )
            return model, None
        elif model_name.lower() == "minigpt4":
            model = BatchGPT4(
                pretrained=model_path,
                batch_size=1,
                trust_remote_code=True,
                device_map="auto",
                #cache_dir=CACHE_DIR,
            )
            return model, None
            
        elif model_name.lower() == "claude":
            model = Claude(
                pretrained=model_path,
                batch_size=1,
                trust_remote_code=True,
                device_map="auto",
                cache_dir=CACHE_DIR,
            )
            return model, None
            
        else:  # Fallback for Molmo using custom wrapper
            print(f"Loading {model_name} from {model_path} with LMMWrapper")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            processor = AutoProcessor.from_pretrained(
                model_path,
                cache_dir=CACHE_DIR,
                trust_remote_code=True,
                device_map="auto",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto",
            ).to(device)
            return model, processor
        
    except Exception as e:
        raise ModelLoadError(f"Error loading {model_name} from {model_path}: {str(e)}") 


import collections
import inspect
import itertools
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from datasets import Image, Sequence
from loguru import logger as eval_logger
from tqdm import tqdm

import lmms_eval.api
import lmms_eval.api.metrics
import lmms_eval.api.registry
from lmms_eval.evaluator_utils import (
    consolidate_group_results,
    consolidate_results,
    get_sample_size,
    get_subtask_list,
    get_task_list,
    prepare_print_tasks,
    print_writeout,
    run_task_tests,
)
from lmms_eval.loggers.evaluation_tracker import EvaluationTracker
from lmms_eval.models import get_model
from lmms_eval.tasks import TaskManager, get_task_dict
from lmms_eval.utils import (
    create_iterator,
    get_datetime_str,
    get_git_commit_hash,
    handle_non_serializable,
    hash_string,
    make_table,
    positional_deprecated,
    run_task_tests,
    simple_parse_args_string,
)


@positional_deprecated
def simple_evaluate(
    model,
    model_args: Optional[Union[str, dict]] = None,
    tasks: Optional[List[Union[str, dict, object]]] = None,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[Union[int, str]] = None,
    max_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: Optional[str] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    delete_requests_cache: bool = False,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = True,
    evaluation_tracker: Optional[EvaluationTracker] = None,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    gen_kwargs: Optional[str] = None,
    task_manager: Optional[TaskManager] = None,
    verbosity: str = "INFO",
    predict_only: bool = False,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    fewshot_random_seed: int = 1234,
    datetime_str: str = get_datetime_str(),
    cli_args=None,
    suffix="",
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str, dict]
        String or dict arguments for each model class, see LM.create_from_arg_string and LM.create_from_arg_object.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        A path to a sqlite db file for caching model responses. `None` if not caching.
    :param cache_requests: bool, optional
        Speed up evaluation by caching the building of dataset requests. `None` if not caching.
    :param rewrite_requests_cache: bool, optional
        Rewrites all of the request cache if set to `True`. `None` if not desired.
    :param delete_requests_cache: bool, optional
        Deletes all of the request cache if set to `True`. `None` if not desired.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderrs. set to 0 for no stderr calculations to be performed.
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param system_instruction: str
        System instruction to be applied to the prompt
    :param apply_chat_template: bool
        If True, apply chat template to the prompt
    :param fewshot_as_multiturn: bool
        Whether to provide the fewshot examples as a multiturn conversation or a single user turn.
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :param predict_only: bool
        If true only model outputs will be generated and returned. Metrics will not be evaluated
    :param random_seed: int
        Random seed for python's random module. If set to None, the seed will not be set.
    :param numpy_random_seed: int
        Random seed for numpy. If set to None, the seed will not be set.
    :param torch_random_seed: int
        Random seed for torch. If set to None, the seed will not be set.
    :param fewshot_random_seed: int
        Random seed for fewshot sampler random generator. If set to None, the seed of generator will be set to None.

    :return
        Dictionary of results
    """
    seed_message = []
    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(torch_random_seed)

    if seed_message:
        eval_logger.info(" | ".join(seed_message))

    assert tasks != [], "No tasks specified, or no tasks found. Please verify the task names."

    if gen_kwargs:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(f"generation_kwargs specified through cli, these settings will be used over set parameters in yaml tasks.")
        if gen_kwargs == "":
            gen_kwargs = None

    if model_args is None:
        model_args = ""

    if task_manager is None:
        task_manager = TaskManager(verbosity, model_name=model)

    task_dict = get_task_dict(tasks, task_manager)

    '''
    ModelClass = get_model(model)
    lm = ModelClass.create_from_arg_string(
        model_args,
        {
            "batch_size": batch_size,
            "device": device,
        },
    )
    '''
    lm = model

    # helper function to recursively apply config overrides to leaf subtasks, skipping their constituent groups.
    # (setting of num_fewshot ; bypassing metric calculation ; setting fewshot seed)
    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }

            else:
                task_obj = task_dict[task_name]
                if type(task_obj) == tuple:
                    group, task_obj = task_obj
                    if task_obj is None:
                        continue
                lm.task_dict[task_name] = task_obj.dataset
                if "generate_until" in task_obj.get_config("output_type"):
                    if gen_kwargs is not None:
                        task_obj.set_config(key="generation_kwargs", value=gen_kwargs, update=True)

                if predict_only:
                    eval_logger.info(f"Processing {task_name} in output-only mode. Metrics will not be calculated!")
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        eval_logger.info(f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored.")
                    else:
                        eval_logger.warning(f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}")
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    # if num_fewshot not provided, and the task does not define a default one, default to 0
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)
                # eval_logger.info(f"Setting fewshot random generator seed to {fewshot_random_seed}")

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    task_dict = _adjust_config(task_dict)

    if check_integrity:
        run_task_tests(task_list=tasks)

    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(
            model_source=model,
            model_args=model_args,
            system_instruction=system_instruction,
            chat_template=lm.chat_template if apply_chat_template else None,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )
    #print(f"task_dict: {task_dict}")
    #print(f"cli_args: {cli_args}")
    #print(f"log_samples: {log_samples}") # True
    #cli_args = {"process_with_media" : True}
    cli_args = SimpleNamespace(process_with_media=True) #,output_path="/n/netscratch/dwork_lab/Lab/katrina/data")
    print(cli_args)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        bootstrap_iters=bootstrap_iters,
        write_out=write_out,
        log_samples=True if predict_only else log_samples,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        verbosity=verbosity,
        cli_args=cli_args,
    )

    if hasattr(lm, "_model"):
        del lm._model
        torch.cuda.empty_cache()

    if lm.rank == 0:
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
        }
        # add more detailed model info if available TODO: add model info
        # if isinstance(lm, lm_eval.models.huggingface.HFLM):
        #     results["config"].update(lm.get_model_info())
        # add info about execution
        results["config"].update(
            {
                "batch_size": batch_size,
                "batch_sizes": (list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []),
                "device": device,
                "use_cache": use_cache,
                "limit": limit,
                "bootstrap_iters": bootstrap_iters,
                "gen_kwargs": gen_kwargs,
                "random_seed": random_seed,
                "numpy_seed": numpy_random_seed,
                "torch_seed": torch_random_seed,
                "fewshot_seed": fewshot_random_seed,
            }
        )
        results["git_hash"] = get_git_commit_hash()
        results["date"] = datetime_str
        # add_env_info(results)  # additional environment info to results
        # add_tokenizer_info(results, lm)  # additional info about tokenizer
        return results
    else:
        return None
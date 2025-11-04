# -*- coding:utf-8 -*-

# Author: Anonymous


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    pipeline,
)
from typing import List, Dict, Tuple, Optional
from utils import *
from torch.utils.checkpoint import checkpoint
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacker import DynamicTemperatureAttacker as DTA

def parse_DTA_args():
    parser = argparse.ArgumentParser(description="Dynamic Target Prompt Attacker")

    parser.add_argument_group(
        "General",
        "General arguments for the attacker.",
    )
    parser.add_argument(
        "--attacker-name",
        type=str,
        default="DTA",
        help="The name of the attacker.",
    )
    parser.add_argument(
        "--local-llm-name-or-path",
        type=str,
        default="/hub/huggingface/models/meta/llama-3-8B-Instruct",
        help="The name or path of the local LLM model (i.e., the target model).",
    )
    parser.add_argument(
        "--local-llm-device",
        type = str, 
        default = "cuda:0",
        help = "The device of the local LLM model (i.e., the target model).",
    )
    parser.add_argument(
        "--use-another-reference-llm",
        action="store_true",
        help="Whether to use another reference LLM model.",
    )
    parser.add_argument(
        "--reference-model-name-or-path",
        type = str, 
        default = "/hub/huggingface/models/meta/llama-3-8B-Instruct",
        help = "The name or path of the reference LLM model. When using the same model as the target model, this argument is ignored.",
    )
    parser.add_argument(
        "--reference-model-device",
        type = str, 
        default = "cuda:0",
        help = "The device of the reference LLM model. When using the same model as the target model, this argument is ignored.",
    )
    parser.add_argument(
        "--judge-llm-name-or-path",
        type=str,
        default="/hub/huggingface/models/hubert233/GPTFuzz/",
        help="The name or path of the judge LLM model.",
    )
    parser.add_argument(
        "--judge-llm-device",
        type = str, 
        default = "cuda:0",
        help = "The device of the judge LLM model.",
    )
    parser.add_argument(
        "--torch-dtype",
        type = str,
        default = "float16",
        help = "The torch dtype of the model. Options: float16, bfloat16, float32.",
    )
    
    parser.add_argument_group(
        "Attack",
        "Arguments for the attack.",
    )
    parser.add_argument(
        "--file-name",
        type = str, 
        default = "../data/raw/DNA_100.csv",
        help = "The name of the file to be attacked.",
    )
    parser.add_argument(
        "--data-start-index",
        type = int,
        default = 0,
        help = "The start index of the data to be attacked.",
    )
    parser.add_argument(
        "--data-end-index",
        type = int,
        default = 100,
        help = "The end index of the data to be attacked.",
    )
    parser.add_argument(
        "--reference-temperature",
        type = float, 
        default = 2.0,
        help = "The temperature of the reference model.",
    )
    parser.add_argument(
        "--reference-num-samples",
        type = int,
        default = 30,
        help = "The number of samples (i.e., responses) to generate from the reference model.",
    )
    parser.add_argument(
        "--response-max-length",
        type = int,
        default = 256,
        help = "The maximum length of the response (i.e., the output) from the reference model.",
    )
    parser.add_argument(
        "--forward-response-length",
        type = int,
        default = 30,
        help = "The maximum length of the response (i.e., the output) from the forward model.",
    )
    
    parser.add_argument(
        "--num-outer-iterations",
        type = int,
        default = 20,
        help = "The number of outer iterations for the attack.",
    )
    
    parser.add_argument(
        "--num-inner-iterations",
        type = int,
        default = 5,
        help = "The number of inner iterations for the attack.",
    )
    
    parser.add_argument(
        "--learning-rate",
        type = float,
        default = 1.5,
        help = "The learning rate for the attack.",
    )
    parser.add_argument(
        "--suffix-length",
        type = int,
        default = 20,
        help = "The length of the suffix to be generated.",
    )
    parser.add_argument(
        "--suffix-topk", 
        type = int, 
        default = 10, 
        help = "The top-k value for the suffix generation.",
    )
    parser.add_argument(
        "--suffix-init-token", 
        type = str, 
        default = "!",
        help = "The initial token for the suffix generation.",
    )
    parser.add_argument(
        "--mask-rej-words",
        action = "store_true",
        help = "Whether to mask the rejected words.",
    )
    parser.add_argument(
        "--save-dir",
        type = str,
        default = "../data/results",
        help = "The directory to save the results.",
    )

    
    return parser.parse_args()



def main():
    
    args = parse_DTA_args()
    
    fn = args.file_name
    start_index = args.data_start_index
    end_index = args.data_end_index
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    local_llm_name_or_path = args.local_llm_name_or_path
    local_model_name = args.local_llm_name_or_path.split("/")[-1]
    local_llm_device = args.local_llm_device
    reference_model_name_or_path = args.reference_model_name_or_path
    reference_model_name = args.reference_model_name_or_path.split("/")[-1]
    reference_model_device = args.reference_model_device
    judge_llm_name_or_path = args.judge_llm_name_or_path
    judge_model_name = args.judge_llm_name_or_path.split("/")[-1]
    judge_llm_device = args.judge_llm_device
    
    reference_model_infer_temperature = args.reference_temperature
    num_ref_infer_samples = args.reference_num_samples
    num_outer_iters = args.num_outer_iterations
    num_inner_iters = args.num_inner_iterations
    learning_rate = args.learning_rate
    suffix_length = args.suffix_length
    suffix_topk = args.suffix_topk
    suffix_init_token = args.suffix_init_token
    response_max_length = args.response_max_length
    forward_response_length = args.forward_response_length
    mask_rej_words = args.mask_rej_words
    torch_dtype = args.torch_dtype
    
    save_path = create_output_filename_and_path(
        save_dir = args.save_dir,
        inupt_filename= fn,
        attacker_name = args.attacker_name,
        local_model_name = local_model_name,
        num_iters = num_outer_iters,
        num_inner_iters = num_inner_iters,
        reference_model_infer_temperature= reference_model_infer_temperature,
        num_ref_infer_samples = num_ref_infer_samples,
        forward_response_length= forward_response_length,
        start_index= start_index,
        end_index= end_index,
    )
    
    print("[INIT] Initializing the attacker...")

    attackers = DTA(
        local_llm_model_name_or_path=local_llm_name_or_path,
        local_llm_device=local_llm_device,
        judge_llm_model_name_or_path=judge_llm_name_or_path,  # ✅ 补上
        judge_llm_device=judge_llm_device,
        ref_local_llm_model_name_or_path=reference_model_name_or_path,  # ✅ 正确顺序
        ref_local_llm_device=reference_model_device,                   # ✅ 正确顺序
        reference_model_infer_temperature=reference_model_infer_temperature,
        num_ref_infer_samples=num_ref_infer_samples,
        dtype=getattr(torch, torch_dtype),  # 确保 dtype 正确
    )
    """
    attackers = DTA(
        local_llm_model_name_or_path=local_llm_name_or_path,
        local_llm_device=local_llm_device,
        ref_local_llm_device=reference_model_name_or_path,
        judge_llm_device=judge_llm_device,
        reference_model_infer_temperature=reference_model_infer_temperature,
        num_ref_infer_samples=num_ref_infer_samples,
        dtype = torch_dtype,
    )
    """
    
    print("[INIT] LOADING THE TARGET DATASET...")
    target_set = load_target_set(fn)
    
    all_results = attackers.attack(
        target_set=target_set,
        num_iters=num_outer_iters,
        num_inner_iters=num_inner_iters,
        learning_rate=learning_rate,
        response_length=response_max_length,
        forward_response_length=forward_response_length,
        suffix_max_length=suffix_length,
        suffix_topk=suffix_topk,
        suffix_init_token=suffix_init_token,
        mask_rejection_words=mask_rej_words,
        verbose=False,
        save_path=save_path,
        start_index = start_index, 
        end_index = end_index,
    )
    
    print("[RESULT] Printing the results...")
    for i, result in enumerate(all_results):
        print(f"Result {i}: {result}")
    
    print("[DONE] The attack is done. The results are saved in the following path:")
    print(f"{save_path}")



if __name__ == "__main__":
    main()
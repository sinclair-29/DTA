# -*- coding:utf-8 -*-
# Author: Anonymous (Reforged by an AI Assistant)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import json
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    pipeline
)
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from peft import get_peft_model, AdaLoraConfig, TaskType
from torch.cuda.amp import autocast, GradScaler

# 配置日志，使其更为专业
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 内部配置类，用于组织参数
@dataclass
class _AttackHyperparams:
    """A data class to hold all attack hyperparameters for clarity."""
    num_outer_iters: int
    num_inner_iters: int
    learning_rate: float
    response_length: int
    forward_response_length: int
    suffix_length: int
    suffix_topk: int
    reference_temp: float
    num_ref_samples: int
    mask_rejection_words: bool


class DynamicTemperatureAttacker:
    """
    A refactored implementation of the Dynamic Temperature Attacker that maintains
    backward compatibility with the original main.py script.
    """

    def __init__(
            self,
            local_llm_model_name_or_path: str,
            local_llm_device: str,
            judge_llm_model_name_or_path: str,
            judge_llm_device: str,
            ref_local_llm_model_name_or_path: Optional[str] = None,
            ref_local_llm_device: Optional[str] = None,
            reference_model_infer_temperature: float = 1.0,
            num_ref_infer_samples: int = 30,
            dtype: torch.dtype = torch.float16,
            **kwargs  # Absorb unused legacy parameters
    ):
        """
        Initializes the attacker. The signature is kept compatible with the old script.
        """
        self.target_llm_device = local_llm_device
        self.ref_llm_device = ref_local_llm_device or local_llm_device
        self.judge_llm_device = judge_llm_device
        self.dtype = dtype

        # 将攻击参数暂存，待 attack 方法调用时再正式设置
        self.reference_temp = reference_model_infer_temperature
        self.num_ref_samples = num_ref_infer_samples

        self._setup_models(
            local_llm_model_name_or_path,
            ref_local_llm_model_name_or_path,
            judge_llm_model_name_or_path
        )
        self.scaler = GradScaler()

    def _setup_models(self, target_path, ref_path, judge_path):
        """Encapsulated logic for loading all models."""
        logger.info("Setting up models...")

        # 1. 目标模型 (Target LLM) with PEFT
        base_target_llm = AutoModelForCausalLM.from_pretrained(target_path, torch_dtype=self.dtype)
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_path)
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=16,
            target_modules=["q_proj", "v_proj"]
        )
        self.target_llm = get_peft_model(base_target_llm, peft_config)
        self.target_llm.to(self.target_llm_device).train()
        logger.info(f"Target LLM '{target_path}' loaded with PEFT and set to train mode.")
        self.target_llm.print_trainable_parameters()

        # 2. 参考模型 (Reference LLM)
        ref_path = ref_path or target_path
        self.ref_llm = AutoModelForCausalLM.from_pretrained(ref_path, torch_dtype=self.dtype)
        self.ref_llm.to(self.ref_llm_device).eval()
        self.ref_tokenizer = AutoTokenizer.from_pretrained(ref_path)
        self.ref_generator = pipeline(
            "text-generation", model=self.ref_llm, tokenizer=self.ref_tokenizer,
            device=self.ref_llm_device
        )
        logger.info(f"Reference LLM '{ref_path}' loaded and set to eval mode.")

        # 3. 评审模型 (Judge LLM)
        self.judge_llm = RobertaForSequenceClassification.from_pretrained(judge_path, torch_dtype=torch.float32)
        self.judge_llm.to(self.judge_llm_device).eval()
        self.judge_tokenizer = RobertaTokenizer.from_pretrained(judge_path)
        logger.info(f"Judge LLM '{judge_path}' loaded and set to eval mode.")

    def _initialize_suffix_logits(self, prompt_ids: torch.Tensor, params: _AttackHyperparams) -> torch.Tensor:
        """Generates initial suffix logits based on the prompt."""
        with torch.no_grad():
            gen_output = self.target_llm.generate(
                input_ids=prompt_ids, max_new_tokens=params.suffix_length,
                do_sample=True, top_k=params.suffix_topk
            )
            full_ids = torch.cat([prompt_ids, gen_output[:, prompt_ids.shape[1]:]], dim=1)
            full_logits = self.target_llm(full_ids).logits
            return full_logits[:, prompt_ids.shape[1] - 1:-1, :].detach()

    def _generate_and_select_reference(self, prompt_embeds: torch.Tensor, suffix_logits: torch.Tensor,
                                       params: _AttackHyperparams) -> Tuple[torch.Tensor, float, str]:
        """Generates candidate responses and selects the best one using the judge model."""
        with torch.no_grad():
            gumbel_probs = F.gumbel_softmax(suffix_logits.float(), tau=0.1, hard=True).to(self.dtype)
            suffix_embeds = gumbel_probs @ self.target_llm.get_input_embeddings().weight
            full_input_embeds = torch.cat([prompt_embeds, suffix_embeds], dim=1).to(self.ref_llm_device)

            ref_ids = self.ref_llm.generate(
                inputs_embeds=full_input_embeds, max_new_tokens=params.response_length,
                num_return_sequences=params.num_ref_samples, do_sample=True, temperature=params.reference_temp
            )
            ref_texts = self.ref_tokenizer.batch_decode(ref_ids[:, full_input_embeds.shape[1]:],
                                                        skip_special_tokens=True)

        best_score, best_text, best_idx = -1.0, "", -1
        for idx, text in enumerate(ref_texts):
            score = self._judge_response(text)[1]  # Unsafe probability
            if score > best_score:
                best_score, best_text, best_idx = score, text, idx

        target_ids = ref_ids[best_idx,
                     full_input_embeds.shape[1]: full_input_embeds.shape[1] + params.forward_response_length]
        return target_ids.unsqueeze(0), best_score, best_text

    def _calculate_loss(self, pred_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Calculates the primary cross-entropy loss."""
        return F.cross_entropy(pred_logits.reshape(-1, pred_logits.size(-1)), target_ids.reshape(-1))

    def _optimize_single_prompt(self, prompt: str, params: _AttackHyperparams) -> Dict[str, Any]:
        """Executes the full optimization loop for a single prompt."""
        prompt_ids = self.target_tokenizer(prompt, return_tensors="pt").input_ids.to(self.target_llm_device)
        prompt_embeds = self.target_llm.get_input_embeddings()(prompt_ids).detach()

        suffix_logits = self._initialize_suffix_logits(prompt_ids, params)
        best_overall_score, best_results = -1.0, {}

        for i in tqdm(range(params.num_outer_iters), desc="Outer Loop"):
            target_ids, ref_score, ref_text = self._generate_and_select_reference(prompt_embeds, suffix_logits, params)
            target_ids = target_ids.to(self.target_llm_device)
            logger.info(f"Outer step {i + 1}: Best ref score: {ref_score:.4f} | Ref text: '{ref_text[:80]}...'")

            noise = torch.zeros_like(suffix_logits, requires_grad=True)
            optimizer = torch.optim.AdamW([noise], lr=params.learning_rate)
            base_logits = suffix_logits.detach()

            for j in tqdm(range(params.num_inner_iters), desc="Inner Loop", leave=False):
                optimizer.zero_grad()
                current_logits = base_logits + noise

                with autocast(enabled=(self.dtype == torch.float16)):
                    gumbel_probs = F.gumbel_softmax(current_logits.float(), tau=0.1, hard=True).to(self.dtype)
                    suffix_embeds = gumbel_probs @ self.target_llm.get_input_embeddings().weight

                    full_embeds = torch.cat(
                        [prompt_embeds, suffix_embeds, self.target_llm.get_input_embeddings()(target_ids)], dim=1)
                    full_logits = self.target_llm(inputs_embeds=full_embeds).logits

                    start_idx = prompt_embeds.shape[1] + suffix_embeds.shape[1]
                    pred_resp_logits = full_logits[:, start_idx - 1:-1, :]

                    loss = self._calculate_loss(pred_resp_logits.float(), target_ids)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

            suffix_logits = (base_logits + noise).detach()

            with torch.no_grad():
                final_suffix_text = self.target_tokenizer.decode(torch.argmax(suffix_logits, dim=-1)[0],
                                                                 skip_special_tokens=True)
                test_output = \
                self.ref_generator(prompt + final_suffix_text, max_new_tokens=params.response_length, do_sample=True)[
                    0]['generated_text']
                test_score = self._judge_response(test_output)[1]
                logger.info(f"Test on '{prompt + final_suffix_text}': Score={test_score:.4f}")

                if test_score > best_overall_score:
                    best_overall_score = test_score
                    best_results = {
                        "prompt": prompt, "suffix": final_suffix_text, "response": test_output,
                        "score": test_score, "reference_response": ref_text, "reference_score": ref_score,
                    }
                    logger.info(f"*** New best score found: {best_overall_score:.4f} ***")

        return best_results or {"prompt": prompt, "status": "failed"}

    def _judge_response(self, response: str) -> List[float]:
        """Judges a response using the judge model."""
        with torch.no_grad():
            inputs = self.judge_tokenizer(response, truncation=True, return_tensors="pt").to(self.judge_llm.device)
            probs = F.softmax(self.judge_llm(**inputs).logits, dim=-1)
            return probs.squeeze().cpu().tolist()

    def attack(
            self,
            target_set: List[str],
            num_iters: int,
            num_inner_iters: int,
            learning_rate: float,
            response_length: int,
            forward_response_length: int,
            suffix_max_length: int,
            suffix_topk: int,
            mask_rejection_words: bool,
            save_path: Optional[str] = None,
            start_index: int = 0,
            end_index: int = 100,
            **kwargs  # Absorb unused legacy parameters
    ) -> List[Dict[str, Any]]:
        """
        Executes the attack on a set of targets. The signature is kept compatible.
        """
        # 将所有攻击参数打包到一个配置对象中，便于内部传递
        attack_params = _AttackHyperparams(
            num_outer_iters=num_iters,
            num_inner_iters=num_inner_iters,
            learning_rate=learning_rate,
            response_length=response_length,
            forward_response_length=forward_response_length,
            suffix_length=suffix_max_length,
            suffix_topk=suffix_topk,
            reference_temp=self.reference_temp,
            num_ref_samples=self.num_ref_samples,
            mask_rejection_words=mask_rejection_words
        )

        prompts_to_attack = target_set[start_index:end_index]
        all_results = []

        try:
            fout = open(save_path, "w", encoding="utf-8") if save_path else None
            for prompt in prompts_to_attack:
                logger.info(f"\n{'=' * 20} Attacking Prompt: '{prompt}' {'=' * 20}")
                result = self._optimize_single_prompt(prompt, attack_params)
                all_results.append(result)
                if fout:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
        finally:
            if fout:
                fout.close()

        return all_results
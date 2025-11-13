# -*- coding:utf-8 -*-
# Author: Anonymous (Reforged by an AI Assistant, v2.2 - Final Fix)
# Version: Optimized for Robustness and Stability

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

try:
    from peft import get_peft_model, AdaLoraConfig, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from torch.cuda.amp import autocast, GradScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class _AttackHyperparams:
    """Dataclass to hold hyperparameters for the attack."""
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
    Implements the Dynamic Temperature Attack.

    This attacker optimizes a malicious suffix to jailbreak a target LLM by using a
    reference LLM to generate desirable (but harmful) responses and then training
    the suffix to elicit similar responses from the target LLM.
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
            **kwargs
    ):
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library is not installed, but is required. Please run 'pip install peft'.")

        self.target_llm_device = local_llm_device
        self.ref_llm_device = ref_local_llm_device or local_llm_device
        self.judge_llm_device = judge_llm_device
        self.dtype = dtype

        self.reference_temp_init = reference_model_infer_temperature
        self.num_ref_samples_init = num_ref_infer_samples

        self._setup_models(
            local_llm_model_name_or_path,
            # If ref_local_llm is not provided, default to the target LLM path.
            ref_local_llm_model_name_or_path or local_llm_model_name_or_path,
            judge_llm_model_name_or_path
        )
        self.scaler = GradScaler()

    def _setup_models(self, target_path: str, ref_path: str, judge_path: str):
        """Loads and configures all required models and tokenizers."""
        logger.info("Setting up models...")

        # --- Target LLM Setup ---
        base_target_llm = AutoModelForCausalLM.from_pretrained(target_path, torch_dtype=self.dtype)
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_path)
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=16,
            target_modules=["q_proj", "v_proj"]
        )
        self.target_llm = get_peft_model(base_target_llm, peft_config)
        self.target_llm.to(self.target_llm_device).train()
        logger.info(f"Target LLM '{target_path}' loaded with PEFT and set to train mode.")
        logger.info(f"Target LLM Vocab Size: {self.target_llm.config.vocab_size}")
        self.target_llm.print_trainable_parameters()

        # --- Reference LLM Setup ---
        self.ref_llm = AutoModelForCausalLM.from_pretrained(ref_path, torch_dtype=self.dtype)
        self.ref_llm.to(self.ref_llm_device).eval()
        logger.info(f"Reference LLM '{ref_path}' loaded and set to eval mode.")
        logger.info(f"Reference LLM Vocab Size: {self.ref_llm.config.vocab_size}")

        # === 优化 1: 健壮的 Tokenizer 处理 ===
        # 如果模型路径相同，直接共享 tokenizer 以确保100%兼容
        if target_path == ref_path:
            logger.info("Target and Reference models are the same. Sharing tokenizer.")
            self.ref_tokenizer = self.target_tokenizer
        else:
            logger.warning("Target and Reference models are different. Loading separate tokenizer for reference model.")
            self.ref_tokenizer = AutoTokenizer.from_pretrained(ref_path)
            # 关键断言: 如果使用不同模型，启动时就检查词汇表大小，提前发现问题
            assert len(self.target_tokenizer) == len(self.ref_tokenizer), \
                f"FATAL: Vocabulary size mismatch! Target: {len(self.target_tokenizer)}, Reference: {len(self.ref_tokenizer)}"

        # 确保所有分词器都有 pad_token，以避免警告和错误
        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token
        if self.ref_tokenizer.pad_token is None:
            self.ref_tokenizer.pad_token = self.ref_tokenizer.eos_token

        self.ref_generator = pipeline(
            "text-generation", model=self.ref_llm, tokenizer=self.ref_tokenizer,
            device=self.ref_llm_device
        )

        # --- Judge LLM Setup ---
        self.judge_llm = RobertaForSequenceClassification.from_pretrained(judge_path, torch_dtype=torch.float32)
        self.judge_llm.to(self.judge_llm_device).eval()
        self.judge_tokenizer = RobertaTokenizer.from_pretrained(judge_path)
        logger.info(f"Judge LLM '{judge_path}' loaded and set to eval mode.")

    def _initialize_suffix_logits(self, prompt_ids: torch.Tensor, params: _AttackHyperparams) -> torch.Tensor:
        with torch.no_grad():
            gen_output = self.target_llm.generate(
                input_ids=prompt_ids, max_new_tokens=params.suffix_length,
                do_sample=True, top_k=params.suffix_topk,
                pad_token_id=self.target_tokenizer.eos_token_id
            )
            full_ids = torch.cat([prompt_ids, gen_output[:, prompt_ids.shape[1]:]], dim=1)
            full_logits = self.target_llm(full_ids).logits
            return full_logits[:, prompt_ids.shape[1] - 1:-1, :].detach()

    def _generate_and_select_reference(self, prompt_embeds: torch.Tensor, suffix_logits: torch.Tensor,
                                       params: _AttackHyperparams) -> Tuple[torch.Tensor, float, str]:
        with torch.no_grad():
            gumbel_probs = F.gumbel_softmax(suffix_logits.float(), tau=0.1, hard=True).to(self.dtype)
            suffix_embeds = gumbel_probs @ self.target_llm.get_input_embeddings().weight
            full_input_embeds = torch.cat([prompt_embeds, suffix_embeds], dim=1).to(self.ref_llm_device)

            ref_ids = self.ref_llm.generate(
                inputs_embeds=full_input_embeds, max_new_tokens=params.response_length,
                num_return_sequences=params.num_ref_samples, do_sample=True, temperature=params.reference_temp,
                pad_token_id=self.ref_tokenizer.eos_token_id
            )
            ref_texts = self.ref_tokenizer.batch_decode(ref_ids[:, full_input_embeds.shape[1]:],
                                                        skip_special_tokens=True)

        best_score, best_text, best_idx = -1.0, "", -1
        for idx, text in enumerate(ref_texts):
            score = self._judge_response(text)[1]
            if score > best_score:
                best_score, best_text, best_idx = score, text, idx

        target_ids = ref_ids[best_idx,
                     full_input_embeds.shape[1]: full_input_embeds.shape[1] + params.forward_response_length]
        return target_ids.unsqueeze(0), best_score, best_text

    def _calculate_loss(self, pred_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        # === 优化 2: 改进损失函数 ===
        # 使用 ignore_index 来忽略填充标记，使损失计算更准确和稳定
        return F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=self.target_tokenizer.pad_token_id
        )

    def _optimize_single_prompt(self, prompt: str, params: _AttackHyperparams) -> Dict[str, Any]:
        prompt_ids = self.target_tokenizer(prompt, return_tensors="pt").input_ids.to(self.target_llm_device)
        prompt_embeds = self.target_llm.get_input_embeddings()(prompt_ids).detach()

        suffix_logits = self._initialize_suffix_logits(prompt_ids, params)
        best_overall_score, best_results = -1.0, {}

        for i in tqdm(range(params.num_outer_iters), desc="Outer Loop"):
            target_ids, ref_score, ref_text = self._generate_and_select_reference(prompt_embeds, suffix_logits, params)
            target_ids = target_ids.to(self.target_llm_device)
            logger.info(f"Outer step {i + 1}: Best ref score: {ref_score:.4f} | Ref text: '{ref_text[:80]}...'")

            # === 优化 3: 运行时安全网 (Runtime Safety Net) ===
            # 在将 ref_llm 生成的 target_ids 用于 target_llm 之前，检查并修正越界的 token ID
            vocab_size = self.target_llm.config.vocab_size
            invalid_mask = (target_ids >= vocab_size) | (target_ids < 0)
            if invalid_mask.any():
                num_invalid = invalid_mask.sum().item()
                logger.warning(
                    f"Found {num_invalid} out-of-bounds token IDs from reference model. "
                    f"Max ID: {target_ids.max().item()}, Vocab size: {vocab_size}. "
                    f"Replacing them with pad_token_id to prevent crashing."
                )
                target_ids[invalid_mask] = self.target_tokenizer.pad_token_id

            # 为确保数值稳定性，在 float32 上创建和优化噪声
            base_logits_float32 = suffix_logits.detach().clone().float()
            noise = torch.zeros_like(base_logits_float32, requires_grad=True)
            optimizer = torch.optim.AdamW([noise], lr=params.learning_rate)

            for j in tqdm(range(params.num_inner_iters), desc="Inner Loop", leave=False):
                optimizer.zero_grad()

                # 在 float32 上应用噪声，以获得更稳定的梯度
                current_logits_float32 = base_logits_float32 + noise

                # 使用 autocast 管理半精度浮点数 (float16) 的计算
                with autocast(enabled=(self.dtype == torch.float16)):
                    # 将 logits 转回模型所需的 dtype，用于 Gumbel-Softmax 和矩阵乘法
                    current_logits_model_dtype = current_logits_float32.to(self.dtype)
                    gumbel_probs = F.gumbel_softmax(current_logits_model_dtype, tau=0.1, hard=True)
                    suffix_embeds = gumbel_probs @ self.target_llm.get_input_embeddings().weight

                    full_embeds = torch.cat(
                        [prompt_embeds, suffix_embeds, self.target_llm.get_input_embeddings()(target_ids)], dim=1)
                    full_logits = self.target_llm(inputs_embeds=full_embeds).logits

                    start_idx = prompt_embeds.shape[1] + suffix_embeds.shape[1]
                    pred_resp_logits = full_logits[:, start_idx - 1:-1, :]

                # 将模型输出转回 float32 进行损失计算，以提高精度和稳定性
                loss = self._calculate_loss(pred_resp_logits.float(), target_ids)

                # GradScaler 自动处理混合精度训练中的梯度缩放
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

            # 从 float32 更新回模型所需的 dtype
            suffix_logits = (base_logits_float32 + noise).detach().to(self.dtype)

            with torch.no_grad():
                final_suffix_text = self.target_tokenizer.decode(torch.argmax(suffix_logits, dim=-1)[0],
                                                                 skip_special_tokens=True)
                test_input = prompt + final_suffix_text
                # 使用 ref_generator (基于 ref_llm) 来评估生成的 suffix 的效果
                generated_output = self.ref_generator(
                    test_input,
                    max_new_tokens=params.response_length,
                    do_sample=True,
                    pad_token_id=self.ref_tokenizer.eos_token_id,
                    num_return_sequences=1
                )[0]
                response_only = generated_output['generated_text'][len(test_input):].strip()
                test_score = self._judge_response(response_only)[1]
                logger.info(f"Test on '{test_input}': Score={test_score:.4f}")

                if test_score > best_overall_score:
                    best_overall_score = test_score
                    best_results = {
                        "prompt": prompt, "suffix": final_suffix_text, "response": response_only,
                        "score": test_score, "reference_response": ref_text, "reference_score": ref_score,
                    }
                    logger.info(f"*** New best score found: {best_overall_score:.4f} ***")

        return best_results or {"prompt": prompt, "status": "failed"}

    def _judge_response(self, response: str) -> List[float]:
        """Judges the harmfulness of a response, returning class probabilities."""
        with torch.no_grad():
            inputs = self.judge_tokenizer(response, truncation=True, max_length=512, return_tensors="pt").to(
                self.judge_llm.device)
            probs = F.softmax(self.judge_llm(**inputs).logits, dim=-1)
            return probs.squeeze().cpu().tolist()

    def attack(self, target_set: List[str], num_iters: int, num_inner_iters: int, learning_rate: float,
               response_length: int, forward_response_length: int, suffix_max_length: int,
               suffix_topk: int, mask_rejection_words: bool, save_path: Optional[str] = None,
               start_index: int = 0, end_index: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Main attack loop over a set of prompts."""
        attack_params = _AttackHyperparams(
            num_outer_iters=num_iters, num_inner_iters=num_inner_iters,
            learning_rate=learning_rate, response_length=response_length,
            forward_response_length=forward_response_length, suffix_length=suffix_max_length,
            suffix_topk=suffix_topk, reference_temp=self.reference_temp_init,
            num_ref_samples=self.num_ref_samples_init, mask_rejection_words=mask_rejection_words
        )

        prompts_to_attack = target_set[start_index:min(end_index, len(target_set))]
        all_results = []

        fout = None
        try:
            if save_path:
                fout = open(save_path, "w", encoding="utf-8")

            for i, prompt in enumerate(prompts_to_attack):
                logger.info(f"\n{'=' * 20} Attacking Prompt {i + 1}/{len(prompts_to_attack)}: '{prompt}' {'=' * 20}")
                result = self._optimize_single_prompt(prompt, attack_params)
                all_results.append(result)
                if fout:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
        except Exception as e:
            logger.error(f"An error occurred during the attack: {e}", exc_info=True)
        finally:
            if fout:
                fout.close()
                logger.info(f"Results saved to {save_path}")

        return all_results
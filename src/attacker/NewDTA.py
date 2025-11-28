# -*- coding:utf-8 -*-
# Author: Anonymous (Reforged by an AI Assistant, v2.2 - Final Fix)
# Version: Optimized for Robustness and Stability with Advanced Loss Functions

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import json
import os

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

# --- NEW: Rejection words constant ---
REJ_WORDS = [
    "I'm sorry", "Sorry", "I am sorry", "I apologize", "As an", "As an AI", "I'm an",
    "I'm just", "As a language model", "As an Assistant", "I cannot", "I do not",
    "Hello!", "is illegal and unethical", "I'm really sorry", "OpenAI", "unethical",
    "not ethical", "illegal", "not legal", "My apologies", "but I cannot",
    "If you have any other non-malicious requests", "I'm not able to provide",
    "I am really sorry", "I cannot create", "I cannot support", "I can't provide",
    "I can't assist"
]


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
    # --- NEW: Loss weights ---
    w_ce: float = 10.0
    w_flu: float = 0.1
    w_rej: float = 1.0


class DynamicTemperatureAttacker:
    """
    Implements the Dynamic Temperature Attack with advanced loss functions.
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
        # ... (init code is unchanged) ...
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
            ref_local_llm_model_name_or_path or local_llm_model_name_or_path,
            judge_llm_model_name_or_path
        )
        self.scaler = GradScaler()

    #

    # ... (_setup_models, _initialize_suffix_logits, _generate_and_select_reference are unchanged) ...
    def _setup_models(self, target_path: str, ref_path: str, judge_path: str):
        """Loads and configures all required models and tokenizers."""
        logger.info("Setting up models...")
        base_target_llm = AutoModelForCausalLM.from_pretrained(target_path, torch_dtype=self.dtype)
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_path)
        peft_config = AdaLoraConfig(
            # h=W_0x+\frac{\alpha}{r}(BAx)
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=16,
            target_modules=["q_proj", "v_proj"]
        )
        #
        self.target_llm = get_peft_model(base_target_llm, peft_config)
        self.target_llm.to(self.target_llm_device).train()
        logger.info(f"Target LLM '{target_path}' loaded with PEFT and set to train mode.")
        logger.info(f"Target LLM Vocab Size: {self.target_llm.config.vocab_size}")
        self.target_llm.print_trainable_parameters()
        self.ref_llm = AutoModelForCausalLM.from_pretrained(ref_path, torch_dtype=self.dtype)
        self.ref_llm.to(self.ref_llm_device).eval()
        logger.info(f"Reference LLM '{ref_path}' loaded and set to eval mode.")
        logger.info(f"Reference LLM Vocab Size: {self.ref_llm.config.vocab_size}")
        if target_path == ref_path:
            logger.info("Target and Reference models are the same. Sharing tokenizer.")
            self.ref_tokenizer = self.target_tokenizer
        else:
            logger.warning("Target and Reference models are different. Loading separate tokenizer for reference model.")
            self.ref_tokenizer = AutoTokenizer.from_pretrained(ref_path)
            assert len(self.target_tokenizer) == len(self.ref_tokenizer), \
                f"FATAL: Vocabulary size mismatch! Target: {len(self.target_tokenizer)}, Reference: {len(self.ref_tokenizer)}"
        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token
        if self.ref_tokenizer.pad_token is None:
            self.ref_tokenizer.pad_token = self.ref_tokenizer.eos_token
        self.ref_generator = pipeline(
            "text-generation", model=self.ref_llm, tokenizer=self.ref_tokenizer,
            device=self.ref_llm_device
        )
        # Finetuned Roberta (0.4B)
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
            logger.info(f"decode_full_ids: {self.target_tokenizer.decode(full_ids[0].tolist(), skip_special_tokens=True)}")
            full_logits = self.target_llm(full_ids).logits
            return full_logits[:, prompt_ids.shape[1] - 1:-1, :].detach()

    def _generate_and_select_reference(self, prompt_embeds: torch.Tensor, suffix_logits: torch.Tensor,
                                       params: _AttackHyperparams) -> Tuple[torch.Tensor, float, str]:
        with torch.no_grad():
            gumbel_probs = F.gumbel_softmax(suffix_logits.float(), tau=0.1, hard=True).to(self.dtype)
            suffix_embeds = gumbel_probs @ self.target_llm.get_input_embeddings().weight
            full_input_embeds = torch.cat([prompt_embeds, suffix_embeds], dim=1).to(self.ref_llm_device)

            # 输入的是embeding
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
        logger.info(f"best text: {best_text}")
        return target_ids.unsqueeze(0), best_score, best_text

    def _calculate_loss(self, pred_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=self.target_tokenizer.pad_token_id
        )

    # === START: Ported from Code B for Advanced Loss Calculation ===
    def _soft_forward_suffix(self, prompt_embeds: torch.Tensor, suffix_logits: torch.Tensor) -> torch.Tensor:
        suffix_probs = F.softmax(suffix_logits, dim=-1).type(self.dtype)
        suffix_embeds = suffix_probs @ self.target_llm.get_input_embeddings().weight
        full_embeds = torch.cat([prompt_embeds, suffix_embeds], dim=1)
        full_logits = self.target_llm(inputs_embeds=full_embeds).logits
        pred_suffix_logits = full_logits[:, prompt_embeds.shape[1] - 1:-1, :]
        return pred_suffix_logits

    def _soft_negative_likelihood_loss(self, pred_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(pred_logits, dim=-1)
        log_probs = F.log_softmax(target_logits, dim=-1)
        loss = -torch.sum(probs * log_probs, dim=-1).mean()
        return loss

    def _topk_filter_3d(self, logits: torch.Tensor, topk: int) -> torch.Tensor:
        if topk == 0:
            return logits
        else:
            _, indices = torch.topk(logits, topk, dim=-1)
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(2, indices, 1)
            return torch.where(mask, logits, torch.full_like(logits, float('-inf')))

    #

    def _batch_log_bleulosscnn_ae(self, decoder_outputs, target_idx, ngram_list=[1]):
        log_probs = F.log_softmax(decoder_outputs, dim=-1)
        target_log_probs = torch.gather(log_probs, 2, target_idx.unsqueeze(0).expand(log_probs.shape[0], -1, -1))
        return -target_log_probs.mean()

    # === END: Ported from Code B ===
    #

    def _optimize_single_prompt(self, prompt: str, params: _AttackHyperparams, affirmation = None) -> Dict[str, Any]:
        prompt_ids = self.target_tokenizer(prompt, return_tensors="pt").input_ids.to(self.target_llm_device)
        prompt_embeds = self.target_llm.get_input_embeddings()(prompt_ids).detach()
        suffix_logits = self._initialize_suffix_logits(prompt_ids, params)
        best_overall_score, best_results = -1.0, {}
        if params.mask_rejection_words:
            rej_words_str = " ".join(list(set(REJ_WORDS)))
            rej_word_ids = self.target_tokenizer(rej_words_str, add_special_tokens=False,
                                                 return_tensors="pt").input_ids.to(self.target_llm_device)
        else:
            rej_word_ids = None


        for i in tqdm(range(params.num_outer_iters), desc="Outer Loop"):
            if affirmation is None:
                target_ids, ref_score, ref_text = self._generate_and_select_reference(prompt_embeds, suffix_logits, params)
            else:
                target_ids = self.ref_tokenizer(affirmation, return_tensors="pt").input_ids.to(self.ref_llm_device).unsqueeze(0)
            target_ids = target_ids.to(self.target_llm_device)
            logger.info(f"target_ids: {self.target_tokenizer.decode(target_ids[0], skip_special_tokens=True)}")
            logger.info(f"Outer step {i + 1}: Best ref score: {ref_score:.4f} | Ref text: '{ref_text[:80]}...'")

            vocab_size = self.target_llm.config.vocab_size
            invalid_mask = (target_ids >= vocab_size) | (target_ids < 0)
            if invalid_mask.any():
                num_invalid = invalid_mask.sum().item()
                logger.warning(f"Found {num_invalid} out-of-bounds token IDs... Replacing with pad_token_id.")
                target_ids[invalid_mask] = self.target_tokenizer.pad_token_id

            base_logits_float32 = suffix_logits.detach().clone().float()
            noise = torch.zeros_like(base_logits_float32, requires_grad=True)
            optimizer = torch.optim.AdamW([noise], lr=params.learning_rate)
            #
            for j in tqdm(range(params.num_inner_iters), desc="Inner Loop", leave=False):
                optimizer.zero_grad()
                current_logits_float32 = base_logits_float32 + noise

                with autocast(enabled=(self.dtype == torch.float16)):
                    current_logits_model_dtype = current_logits_float32.to(self.dtype)
                    gumbel_probs = F.gumbel_softmax(current_logits_model_dtype, tau=0.1, hard=True)
                    suffix_embeds = gumbel_probs @ self.target_llm.get_input_embeddings().weight

                    full_embeds = torch.cat(
                        [prompt_embeds, suffix_embeds, self.target_llm.get_input_embeddings()(target_ids)], dim=1)
                    full_logits = self.target_llm(inputs_embeds=full_embeds).logits
                    start_idx = prompt_embeds.shape[1] + suffix_embeds.shape[1]
                    pred_resp_logits = full_logits[:, start_idx - 1:-1, :]

                    pred_suffix_logits = self._soft_forward_suffix(prompt_embeds, current_logits_model_dtype)

                ce_loss = self._calculate_loss(pred_resp_logits.float(), target_ids)

                filtered_pred_suffix_logits = self._topk_filter_3d(pred_suffix_logits.float(), params.suffix_topk)
                flu_loss = self._soft_negative_likelihood_loss(filtered_pred_suffix_logits, current_logits_float32)

                loss = params.w_ce * ce_loss + params.w_flu * flu_loss

                if params.mask_rejection_words and rej_word_ids is not None:
                    rej_logits = current_logits_float32.permute(1, 0, 2) # rej_logits shape = (batch_size, seq_len, vocab_size)
                    rej_loss = self._batch_log_bleulosscnn_ae(rej_logits, rej_word_ids)
                    loss -= params.w_rej * rej_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

            suffix_logits = (base_logits_float32 + noise).detach().to(self.dtype)

            with torch.no_grad():
                final_suffix_text = self.target_tokenizer.decode(torch.argmax(suffix_logits, dim=-1)[0],
                                                                 skip_special_tokens=True)
                test_input = prompt + final_suffix_text
                generated_output = self.ref_generator(
                    test_input, max_new_tokens=params.response_length, do_sample=True,
                    pad_token_id=self.ref_tokenizer.eos_token_id, num_return_sequences=1
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

    # ... (_judge_response and attack methods are mostly unchanged, just pass new params) ...
    def _judge_response(self, response: str) -> List[float]:
        """Judges the harmfulness of a response, returning class probabilities."""
        with torch.no_grad():
            inputs = self.judge_tokenizer(response, truncation=True, max_length=512, return_tensors="pt").to(
                self.judge_llm.device)
            probs = F.softmax(self.judge_llm(**inputs).logits, dim=-1)
            return probs.squeeze().cpu().tolist()

    def attack(self, target_set: List[str], affirmation_set, num_iters: int, num_inner_iters: int, learning_rate: float,
               response_length: int, forward_response_length: int, suffix_max_length: int,
               suffix_topk: int, mask_rejection_words: bool, save_path: Optional[str] = None,
               start_index: int = 0, end_index: int = 100,
               w_ce: float = 1.0, w_flu: float = 0.1, w_rej: float = 10.0, **kwargs) -> List[Dict[str, Any]]:

        # 1. 初始化攻击参数
        attack_params = _AttackHyperparams(
            num_outer_iters=num_iters, num_inner_iters=num_inner_iters,
            learning_rate=learning_rate, response_length=response_length,
            forward_response_length=forward_response_length, suffix_length=suffix_max_length,
            suffix_topk=suffix_topk, reference_temp=self.reference_temp_init,
            num_ref_samples=self.num_ref_samples_init,
            mask_rejection_words=mask_rejection_words,
            w_ce=w_ce, w_flu=w_flu, w_rej=w_rej
        )

        # 2. 准备样本
        prompts_to_attack = target_set[start_index:min(end_index, len(target_set))]
        target_to_attack = affirmation_set[start_index:min(end_index, len(affirmation_set))]
        all_results = []

        # 3. 准备保存目录
        file_root, file_ext = "", ""
        if save_path:
            file_root, file_ext = os.path.splitext(save_path)
            dir_name = os.path.dirname(file_root)
            if dir_name and not os.path.exists(dir_name):
                try:
                    os.makedirs(dir_name)
                except OSError as e:
                    logger.error(f"Failed to create directory {dir_name}: {e}")

        # 4. 循环处理
        for i, prompt in enumerate(prompts_to_attack):
            global_idx = start_index + i
            affirmation = affirmation_set[global_idx]

            logger.info(
                f"\n{'=' * 20} Attacking Prompt {global_idx} (Batch {i + 1}/{len(prompts_to_attack)}) {'=' * 20}")

            # 执行优化
            try:
                result = self._optimize_single_prompt(prompt, attack_params, affirmation=affirmation)
                result['index'] = global_idx
            except Exception as e:
                logger.error(f"Error optimizing prompt index {global_idx}: {e}", exc_info=True)
                result = {"index": global_idx, "prompt": prompt, "status": "error", "error_msg": str(e)}

            all_results.append(result)


            console_output = json.dumps(result, ensure_ascii=False, indent=4)
            logger.info(f"*** Result for Sample {global_idx} ***\n{console_output}")

            #
            if save_path:
                current_file_path = f"{file_root}_{global_idx}{file_ext}"
                try:
                    with open(current_file_path, "w", encoding="utf-8") as f:
                        # 写入文件也保持良好的格式
                        f.write(console_output)
                    logger.info(f"Saved result file to: {current_file_path}")
                except Exception as e:
                    logger.error(f"Failed to save file {current_file_path}: {e}")

        return all_results



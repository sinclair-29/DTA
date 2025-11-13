# -*- coding:utf-8 -*-
# Author: Anonymous (Reforged by an AI Assistant, v2.1)

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

# 尝试导入PEFT库，若失败则在需要时抛出错误
try:
    from peft import get_peft_model, AdaLoraConfig, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from torch.cuda.amp import autocast, GradScaler

# 配置日志系统，使其输出格式清晰，便于调试
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class _AttackHyperparams:
    """
    一个内部使用的数据类，用于整洁地组织和传递所有攻击相关的超参数。
    """
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
    重铸版的 Dynamic Temperature Attacker。
    此实现保持了与旧版 main.py 脚本的接口兼容性，
    同时内部结构经过现代化重构，提升了可读性、可维护性，并修复了已知的bug。
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
            **kwargs  # 安全地吸收并忽略旧接口中任何未使用的参数
    ):
        """
        初始化攻击器。方法签名与旧脚本完全兼容。
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library is not installed, but is required. Please run 'pip install peft'.")

        self.target_llm_device = local_llm_device
        self.ref_llm_device = ref_local_llm_device or local_llm_device
        self.judge_llm_device = judge_llm_device
        self.dtype = dtype

        # 将部分初始化参数暂存，待 attack 方法调用时打包
        self.reference_temp_init = reference_model_infer_temperature
        self.num_ref_samples_init = num_ref_infer_samples

        self._setup_models(
            local_llm_model_name_or_path,
            ref_local_llm_model_name_or_path,
            judge_llm_model_name_or_path
        )
        self.scaler = GradScaler()

    def _setup_models(self, target_path: str, ref_path: Optional[str], judge_path: str):
        """将所有模型的加载、配置和设备分配逻辑封装于此。"""
        logger.info("Setting up models...")

        # 1. 配置并加载带PEFT的目标模型 (Target LLM)
        base_target_llm = AutoModelForCausalLM.from_pretrained(target_path, torch_dtype=self.dtype)
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_path)
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=16,
            target_modules=["q_proj", "v_proj"]
        )
        self.target_llm = get_peft_model(base_target_llm, peft_config)
        self.target_llm.to(self.target_llm_device).train()  # 设为训练模式以更新适配器
        logger.info(f"Target LLM '{target_path}' loaded with PEFT and set to train mode.")
        self.target_llm.print_trainable_parameters()

        # 2. 配置并加载参考模型 (Reference LLM)
        ref_path = ref_path or target_path
        self.ref_llm = AutoModelForCausalLM.from_pretrained(ref_path, torch_dtype=self.dtype)
        self.ref_llm.to(self.ref_llm_device).eval()  # 仅用于推理，设为评估模式
        self.ref_tokenizer = AutoTokenizer.from_pretrained(ref_path)
        self.ref_generator = pipeline(
            "text-generation", model=self.ref_llm, tokenizer=self.ref_tokenizer,
            device=self.ref_llm_device
        )
        logger.info(f"Reference LLM '{ref_path}' loaded and set to eval mode.")

        # 3. 配置并加载评审模型 (Judge LLM)
        self.judge_llm = RobertaForSequenceClassification.from_pretrained(judge_path, torch_dtype=torch.float32)
        self.judge_llm.to(self.judge_llm_device).eval()  # 仅用于推理，设为评估模式
        self.judge_tokenizer = RobertaTokenizer.from_pretrained(judge_path)
        logger.info(f"Judge LLM '{judge_path}' loaded and set to eval mode.")

    def _initialize_suffix_logits(self, prompt_ids: torch.Tensor, params: _AttackHyperparams) -> torch.Tensor:
        """根据prompt生成初始的后缀logits。"""
        with torch.no_grad():
            gen_output = self.target_llm.generate(
                input_ids=prompt_ids, max_new_tokens=params.suffix_length,
                do_sample=True, top_k=params.suffix_topk
            )
            # 为了获取logits，我们需要重新进行一次前向传播
            full_ids = torch.cat([prompt_ids, gen_output[:, prompt_ids.shape[1]:]], dim=1)
            full_logits = self.target_llm(full_ids).logits
            # 截取对应后缀部分的logits
            return full_logits[:, prompt_ids.shape[1] - 1:-1, :].detach()

    def _generate_and_select_reference(self, prompt_embeds: torch.Tensor, suffix_logits: torch.Tensor,
                                       params: _AttackHyperparams) -> Tuple[torch.Tensor, float, str]:
        """生成候选回复，并使用评审模型选出最佳的一个作为优化目标。"""
        with torch.no_grad():
            gumbel_probs = F.gumbel_softmax(suffix_logits.float(), tau=0.1, hard=True).to(self.dtype)
            suffix_embeds = gumbel_probs @ self.target_llm.get_input_embeddings().weight
            full_input_embeds = torch.cat([prompt_embeds, suffix_embeds], dim=1).to(self.ref_llm_device)

            ref_ids = self.ref_llm.generate(
                inputs_embeds=full_input_embeds, max_new_tokens=params.response_length,
                num_return_sequences=params.num_ref_samples, do_sample=True, temperature=params.reference_temp,
                pad_token_id=self.ref_tokenizer.eos_token_id  # 避免生成过程中的警告
            )
            ref_texts = self.ref_tokenizer.batch_decode(ref_ids[:, full_input_embeds.shape[1]:],
                                                        skip_special_tokens=True)

        best_score, best_text, best_idx = -1.0, "", -1
        for idx, text in enumerate(ref_texts):
            score = self._judge_response(text)[1]  # 取"不安全"的概率
            if score > best_score:
                best_score, best_text, best_idx = score, text, idx

        target_ids = ref_ids[best_idx,
                     full_input_embeds.shape[1]: full_input_embeds.shape[1] + params.forward_response_length]
        return target_ids.unsqueeze(0), best_score, best_text

    def _calculate_loss(self, pred_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        计算交叉熵损失。
        关键修正：将输入logits明确转换为float32，以确保与GradScaler的兼容性。
        """
        return F.cross_entropy(
            pred_logits.float().reshape(-1, pred_logits.size(-1)),
            target_ids.reshape(-1)
        )

    def _optimize_single_prompt(self, prompt: str, params: _AttackHyperparams) -> Dict[str, Any]:
        """为单个prompt执行完整的双循环优化过程。"""
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

                    # 将prompt, suffix和目标回复拼接成完整的输入序列以进行前向传播
                    full_embeds = torch.cat(
                        [prompt_embeds, suffix_embeds, self.target_llm.get_input_embeddings()(target_ids)], dim=1)
                    full_logits = self.target_llm(inputs_embeds=full_embeds).logits

                    # 截取与目标回复对应的logits部分
                    start_idx = prompt_embeds.shape[1] + suffix_embeds.shape[1]
                    pred_resp_logits = full_logits[:, start_idx - 1:-1, :]

                    loss = self._calculate_loss(pred_resp_logits, target_ids)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

            suffix_logits = (base_logits + noise).detach()

            with torch.no_grad():
                final_suffix_text = self.target_tokenizer.decode(torch.argmax(suffix_logits, dim=-1)[0],
                                                                 skip_special_tokens=True)
                test_input = prompt + final_suffix_text
                test_output = self.ref_generator(test_input, max_new_tokens=params.response_length, do_sample=True)[0][
                    'generated_text']
                # 从完整输出中提取生成的回复部分
                response_only = test_output[len(test_input):].strip()
                test_score = self._judge_response(response_only)[1]
                logger.info(f"Test on '{test_input}': Score={test_score:.4f}")

                if test_score > best_overall_score:
                    best_overall_score = test_score
                    best_results = {
                        "prompt": prompt, "suffix": final_suffix_text, "response": test_output,
                        "score": test_score, "reference_response": ref_text, "reference_score": ref_score,
                    }
                    logger.info(f"*** New best score found: {best_overall_score:.4f} ***")

        return best_results or {"prompt": prompt, "status": "failed"}

    def _judge_response(self, response: str) -> List[float]:
        """使用评审模型评估一个回复的安全性。"""
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
            **kwargs  # 兼容旧接口，吸收未使用的参数
    ) -> List[Dict[str, Any]]:
        """
        在目标集上执行攻击。方法签名与旧脚本完全兼容。
        """
        # 将所有攻击参数打包到一个配置对象中，使内部调用更整洁
        attack_params = _AttackHyperparams(
            num_outer_iters=num_iters,
            num_inner_iters=num_inner_iters,
            learning_rate=learning_rate,
            response_length=response_length,
            forward_response_length=forward_response_length,
            suffix_length=suffix_max_length,
            suffix_topk=suffix_topk,
            reference_temp=self.reference_temp_init,
            num_ref_samples=self.num_ref_samples_init,
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
                    fout.flush()  # 确保实时写入文件
        finally:
            if fout:
                fout.close()

        return all_results
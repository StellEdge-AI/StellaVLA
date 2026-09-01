# Copyright 2025 stellavla community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import warnings
from typing import Optional

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast


from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 151669 # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = 153716


import torch.nn as nn


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")

        # flash_attention_2 requires the flash-attn package (not on ROCm, and
        # not always installed on CUDA boxes either — e.g. some CUDA hosts'
        # stellavla env). Fall back to sdpa when flash_attn isn't importable.
        import torch.backends
        if not torch.cuda.is_available() or torch.version.hip:
            _attn_impl = "sdpa"
        else:
            try:
                import flash_attn  # noqa: F401
                _attn_impl = "flash_attention_2"
            except ImportError:
                _attn_impl = "sdpa"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=_attn_impl,
            dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # Optional max_seq_len cap for apply_chat_template — guards against
        # outlier-long CoT prompts that could OOM the batch. Default None
        # (no truncation, preserves prior behavior). For LIBERO StellaVLA full
        # mode we set 1536 in the yaml (measured max=1328, ~200 token margin).
        self.max_seq_len = qwenvl_config.get("max_seq_len", None)
        if self.max_seq_len is not None:
            print(f"[QWen3VL] max_seq_len={self.max_seq_len} "
                  f"(truncation=longest_first), padding_side={processor.tokenizer.padding_side}")

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        # If max_seq_len is set, also enable truncation. With padding_side="left"
        # the truncation will drop tokens from the LEFT (oldest content), keeping
        # the action 🔍 tokens at the end intact — critical for the OFT pattern
        # (hidden states at action-token positions feed the MLP head).
        _kwargs = {}
        if self.max_seq_len is not None:
            _kwargs["max_length"] = int(self.max_seq_len)
            _kwargs["truncation"] = True
        batch_inputs = self.processor.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **_kwargs,
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None: #  here only for fast_tokenizer now. 
            action_token_min = _ACTION_TOKEN_MIN # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX
            labels = batch_inputs['input_ids'].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    warnings.warn("no action token in this sequence; the whole "
                                  "sequence was masked out of the language loss")
            
            labels[labels == self.processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
            batch_inputs['labels'] = labels

        return batch_inputs.to(self.model.device)

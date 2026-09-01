# Copyright 2026 stellavla community.
# Licensed under the MIT License.
"""StellaVLA — Qwen3-VL backbone + OpenVLA-OFT MLP/L1 action head.

One shared Qwen3-VL backbone reads
`[retrieved demo | current observation | task instruction]`. Two experts read
its hidden states: a language expert (trained, not deployed) and an action
expert (trained and deployed).

The action expert reads `chunk_len` action-placeholder tokens (🔍) appended to
the **user** turn. Because they sit causally *before* the assistant's language
target, that target can never leak into the action queries, so the action path
stays valid with language decoding switched off — which is how it is deployed.
Inference is one VLM forward, a gather at the 🔍 positions, and one MLP
regression: no autoregressive decode and no denoising loop.

State reaches the model only as the `state_text` line in the prompt; there is no
separate numeric state encoder.

This is an inference-only build of the framework. Training code lives elsewhere.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import torch
from qwen_vl_utils import process_vision_info
from torch.nn.utils.rnn import pad_sequence

from deployment.model_server.tools.image_tools import to_pil_preserve
from stellavla.model.framework.base_framework import baseframework
from stellavla.model.modules.action_model.MLP_ActionHeader import (
    get_action_model as get_mlp_action_model,
)
from stellavla.model.modules.vlm import get_vlm_model
from stellavla.model.tools import FRAMEWORK_REGISTRY
from stellavla.utils import initialize_overwatch, resize_images

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IM_END = "<|im_end|>"


@FRAMEWORK_REGISTRY.register("StellaVLA")
class StellaVLA(baseframework):
    """Qwen3-VL + OFT MLP/L1 action head, wired for evaluation."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config

        # VLM: loaded from framework.qwenvl.base_vlm, brings its own tokenizer
        # and chat template.
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        vl_hidden = self.qwen_vl_interface.model.config.hidden_size

        # Prompt shape. These must match how the checkpoint was trained: the
        # tokenizer below builds the demo and wrist branches from them.
        fw = config.framework
        self.use_context_demo = bool(fw.get("use_context_demo", False))
        self.include_wrist_image = bool(fw.get("include_wrist_image", False))
        self.output_schema = str(fw.get("output_schema", "movement_only"))
        if self.output_schema not in (
            "full", "movement_only", "movement_pure", "movement_no3d", "movement_no2d"
        ):
            raise ValueError(
                "output_schema must be 'full', 'movement_only', 'movement_pure', "
                f"'movement_no3d' or 'movement_no2d', got {self.output_schema!r}"
            )
        self.lang_w = float(fw.get("language_loss_weight", 0.3))

        am = fw.action_model
        self.future_action_window_size = int(am.future_action_window_size)
        self.past_action_window_size = int(am.past_action_window_size)
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # OFT MLP/L1 head, sized to the VLM hidden dim.
        self.config.framework.action_model.action_hidden_dim = vl_hidden
        self.action_model = get_mlp_action_model(config=self.config)

        # Action placeholder token.
        self.action_token = "🔍"
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer(
            "🔍", add_special_tokens=False
        )["input_ids"][0]

        logger.info(
            f"[StellaVLA] OFT MLP/L1 head; action_token={self.action_token!r} "
            f"action_token_id={self.action_token_id} chunk_len={self.chunk_len} "
            f"hidden={vl_hidden} use_context_demo={self.use_context_demo} "
            f"include_wrist_image={self.include_wrist_image} "
            f"output_schema={self.output_schema} (state via prompt text only; "
            f"trained with lang_w={self.lang_w})"
        )

    # ---------------------------------------------------------------- prompt
    def _extra_user_suffix_text(self) -> str:
        # Cannot put a space between the 🔍 glyphs or they may tokenize apart.
        action_tokens = self.action_token * self.chunk_len
        return (
            f" Please predict the next {self.chunk_len} robot actions: "
            f"<action>{action_tokens}<action>."
        )

    def _gather_action_token_embeddings(
        self, last_hidden: torch.Tensor, input_ids: torch.Tensor, action_token_id
    ) -> torch.Tensor:
        """Gather the last `chunk_len` action-token hidden states → (B, chunk_len, H).
"""
        device = input_ids.device
        B, L, H = last_hidden.shape
        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)
        counts = mask.sum(dim=1)
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"[StellaVLA] action-token count < {self.chunk_len} for samples "
                f"{insufficient} | counts={counts.tolist()} | action_token_id={action_token_id}"
            )
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)
        return last_hidden.gather(dim=1, index=expanded_index)

    def _build_inference_inputs(
        self, examples: List[dict]
    ) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        """Build inference-time tokenized inputs (no assistant target, no labels)."""
        processor = self.qwen_vl_interface.processor
        tok = processor.tokenizer
        pad_id = tok.pad_token_id

        per_sample = []
        for ex in examples:
            per_sample.append(self._tokenize_one(ex, processor, tok, with_target=False))

        input_ids = pad_sequence(
            [s["input_ids"] for s in per_sample], batch_first=True, padding_value=pad_id
        )
        attention_mask = pad_sequence(
            [s["attention_mask"] for s in per_sample], batch_first=True, padding_value=0
        )
        # No langact at inference; prompt_mask = attention_mask
        langact_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        prompt_mask = attention_mask.bool()

        pixel_values = torch.cat([s["pixel_values"] for s in per_sample], dim=0)
        image_grid_thw = torch.cat([s["image_grid_thw"] for s in per_sample], dim=0)

        qwen_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        if per_sample[0].get("mm_token_type_ids") is not None:
            qwen_inputs["mm_token_type_ids"] = pad_sequence(
                [s["mm_token_type_ids"] for s in per_sample],
                batch_first=True, padding_value=0,
            )
        return qwen_inputs, langact_mask, prompt_mask
    def _tokenize_one(self, ex: dict, processor, tok, *, with_target: bool,
                      prompt_only: bool = False):
        """Tokenize one sample.

        Mirrors the training-time construction (ctxdemo + wrist branches +
        movement_only schema). The chat-template + image-token-expansion path is
        identical, so models trained with that trainer load byte-compatibly.
        """
        # ── Extract per-sample fields ──
        sample_images = ex["image"] if isinstance(ex["image"], list) else [ex["image"]]
        image = sample_images[0]
        wrist_image = sample_images[1] if (self.include_wrist_image and len(sample_images) > 1) else None
        instruction = ex["lang"]
        planning_text = ex.get("planning_text", "") if with_target else ""
        context_demo = ex.get("context_demo") if self.use_context_demo else None
        state_text = ex.get("state_text", "")  # current-obs eef state (language)
        # gripper_2d prompt consistency: ask for "Steering Command 2D" iff THIS
        # sample's target has it (gripper_2d present AND not dropped, flagged by the
        # dataloader). Default True preserves behavior for models/evals predating
        # this flag. Used to build the Movement Command line + demo 2D reference.
        has_gripper_2d = bool(ex.get("has_gripper_2d", True))
        mv_2d_suffix = " and Steering Command 2D (pixel path)" if has_gripper_2d else ""
        demo_2d_ref = " and 2D pixel trace" if has_gripper_2d else ""
        # Output-template ablation variants (asks mirror the dataloader targets):
        #   movement_pure  → drop the auxiliary "Subtask:" ask line
        #   movement_no3d  → drop the 3D clause from the Movement ask
        _subtask_ask = ("" if self.output_schema == "movement_pure"
                        else "Subtask: the subtask you are currently executing\n")
        _subtask_ask_demo = ("" if self.output_schema == "movement_pure"
                             else "Subtask: the subtask currently being executed "
                                  "(identified among the reference demo subgoals)\n")
        if self.output_schema == "movement_no3d":
            _mv_ask = "Movement Command: Steering Command 2D (pixel path)"
            _demo_motion_ref = "demonstrated 2D pixel trace"
        elif self.output_schema == "movement_no2d":
            _mv_ask = "Movement Command: Steering Command 3D (mm/deg motion)"
            _demo_motion_ref = "demonstrated 3D motion"
        else:
            _mv_ask = f"Movement Command: Steering Command 3D (mm/deg motion){mv_2d_suffix}"
            _demo_motion_ref = f"demonstrated 3D motion{demo_2d_ref}"
        demo_frame_states = (context_demo.get("demo_frame_states", [])
                             if context_demo else [])
        demo_initial_state = context_demo.get("demo_initial_state", "") if context_demo else ""
        demo_final_state = context_demo.get("demo_final_state", "") if context_demo else ""

        # ── View intros ──
        if wrist_image is not None:
            view_intro_motion = (
                "You are a robot motion planner. Two camera views are provided: "
                "a third-person view and a supplementary wrist view. Observe both "
                "images and emit the next-second motion command for a low-level "
                "robot controller."
            )
            view_intro_planner = (
                "You are a robot task planner. Two camera views are provided: "
                "a third-person view and a supplementary wrist view. Observe both "
                "images and produce a structured action plan that guides a "
                "low-level robot controller."
            )
            view_intro_ctxdemo = (
                "You are a robot motion planner. Two camera views are provided "
                "above (third-person and supplementary wrist). Emit the next-second "
                "motion command for a low-level robot controller. Use the reference "
                "demonstration to inform your motion command — the subtask-level "
                "motions in the demo show how a similar task is executed."
            )
        else:
            view_intro_motion = (
                "You are a robot motion planner. Observe the workspace image and "
                "emit the next-second motion command for a low-level robot controller."
            )
            view_intro_planner = (
                "You are a robot task planner. Observe the workspace image and "
                "produce a structured action plan that guides a low-level robot controller."
            )
            view_intro_ctxdemo = (
                "You are a robot motion planner. Observe the workspace image above "
                "and emit the next-second motion command for a low-level robot "
                "controller. Use the reference demonstration to inform your motion "
                "command — the subtask-level motions in the demo show how a similar "
                "task is executed."
            )

        # ── Build user message content ──
        if self.output_schema in ("movement_only", "movement_pure", "movement_no3d", "movement_no2d"):
            base_user_text = (
                f"{view_intro_motion}\n\n"
                f"Task: {instruction}\n\n"
                "Respond strictly in the following structure:\n"
                f"{_subtask_ask}"
                f"{_mv_ask}"
            )
        else:
            base_user_text = (
                f"{view_intro_planner}\n\n"
                f"Task: {instruction}\n\n"
                "Respond strictly in the following structure:\n"
                "Task Plan: numbered subtask steps\n"
                "Subtask: the step currently being executed, with Reason explaining spatial context\n"
                f"Movement Command: Steering Command 3D (mm/deg motion){mv_2d_suffix}"
            )

        # Gate accepts image-bearing OR text-only demos (text-only occurs only in
        # the eval-side --demo_strip image ablation; training payloads always
        # carry frames, so this is a no-op for training).
        if self.use_context_demo and context_demo is not None and (
            context_demo.get("demo_frames") or context_demo.get("demo_subtask_summary")
        ):
            demo_text = (
                "Reference demonstration of a similar task:\n"
                f"Demo Task: {context_demo['demo_instruction']}\n\n"
                f"Demo Task Plan:\n{context_demo['demo_task_plan']}\n\n"
                + (f"Demo initial robot state: {demo_initial_state}\n" if demo_initial_state else "")
                + (f"Demo final robot state: {demo_final_state}\n" if demo_final_state else "")
                + ("\nDemo Subgoal Frames (in order, one image per subtask end-state, "
                   "each annotated with the robot state at that subgoal):\n"
                   if context_demo.get("demo_frames") else "")
            )
            demo_summary_text = (
                f"\nDemo Subtask Motions (one row per subtask, in execution order):\n"
                f"{context_demo['demo_subtask_summary']}\n\n"
                "---\n"
            )
            current_user_text = (
                f"{view_intro_ctxdemo}\n\n"
                f"Task: {instruction}\n\n"
                "Ground your prediction in the reference demonstration: match the current view "
                "and current robot state against the demo subgoal images and their robot states to "
                "locate which subtask is currently being executed, then use that subtask's "
                f"{_demo_motion_ref} to inform your Steering Commands.\n"
                "Respond strictly in the following structure:\n"
                f"{_subtask_ask_demo}"
                f"{_mv_ask}"
            )
            content = [{"type": "text", "text": demo_text}]
            # Each demo subgoal frame (= a subtask end-state) annotated with the
            # robot state at that subgoal, so the model can read state→motion correspondence.
            for di, demo_frame in enumerate(context_demo["demo_frames"]):
                content.append({"type": "image", "image": demo_frame})
                dst = demo_frame_states[di] if di < len(demo_frame_states) else ""
                if dst:
                    content.append({"type": "text", "text": f"Demo subgoal {di + 1} state: {dst}"})
            content.append({"type": "text", "text": demo_summary_text})
            content.append({"type": "image", "image": image})
            if wrist_image is not None:
                content.append({"type": "image", "image": wrist_image})
            if state_text:
                content.append({"type": "text", "text": f"Current robot state: {state_text}"})
            content.append({"type": "text", "text": current_user_text})
            user_msgs = [{"role": "user", "content": content}]
        else:
            content = [{"type": "image", "image": image}]
            if wrist_image is not None:
                content.append({"type": "image", "image": wrist_image})
            if state_text:
                content.append({"type": "text", "text": f"Current robot state: {state_text}"})
            content.append({"type": "text", "text": base_user_text})
            user_msgs = [{"role": "user", "content": content}]

        # Optional subclass hook: append extra text to the END of the user turn
        # (e.g. OFT action-placeholder tokens). `content` is the same list object
        # referenced by `user_msgs`, so appending here is reflected. Placing it in
        # the user turn keeps it causally BEFORE the assistant langact target →
        # knowledge isolation is preserved.
        extra_suffix = self._extra_user_suffix_text()
        if extra_suffix:
            content.append({"type": "text", "text": extra_suffix})

        # ── Tokenize prompt (with add_generation_prompt=True) ──
        prompt_text = processor.apply_chat_template(
            user_msgs, tokenize=False, add_generation_prompt=True
        )
        if prompt_only:
            # Failure-viz: return the assembled ctxdemo prompt STRING (image
            # placeholders left unexpanded -> readable panel). Skips VLM tensorization.
            return prompt_text
        img_inputs, _ = process_vision_info(user_msgs)
        prompt_inputs = processor(
            text=[prompt_text],
            images=img_inputs,
            padding=False,
            return_tensors="pt",
        )
        prompt_ids = prompt_inputs["input_ids"].squeeze(0)
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]
        # transformers>=5.5 Qwen3-VL processors return mm_token_type_ids
        # (0=text, 1=image) and the model hard-requires it for M-RoPE whenever
        # image_grid_thw is passed. Thread it through when present; older
        # processors don't return it -> key absent, behavior unchanged.
        mm_prompt = prompt_inputs.get("mm_token_type_ids")
        if mm_prompt is not None:
            mm_prompt = mm_prompt.squeeze(0)

        if not with_target:
            # Inference: no assistant target appended
            input_ids = prompt_ids
            attention_mask = torch.ones(len(input_ids), dtype=torch.long)
            labels = torch.full_like(input_ids, IGNORE_INDEX)
            langact_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            ret = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "langact_mask": langact_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
            if mm_prompt is not None:
                ret["mm_token_type_ids"] = mm_prompt
            return ret

        # ── Append assistant response (planning_text + <|im_end|>\n) ──
        response_str = f"{planning_text}{IM_END}\n"
        resp_ids = tok(
            response_str, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(0)

        input_ids = torch.cat([prompt_ids, resp_ids], dim=0)
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        labels = torch.cat(
            [torch.full((len(prompt_ids),), IGNORE_INDEX, dtype=torch.long), resp_ids],
            dim=0,
        )
        langact_mask = torch.cat(
            [
                torch.zeros(len(prompt_ids), dtype=torch.bool),
                torch.ones(len(resp_ids), dtype=torch.bool),
            ],
            dim=0,
        )
        ret = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "langact_mask": langact_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        if mm_prompt is not None:
            # response tokens are text-only -> type 0
            ret["mm_token_type_ids"] = torch.cat(
                [mm_prompt, torch.zeros(len(resp_ids), dtype=mm_prompt.dtype)], dim=0
            )
        return ret

    # ---------------------------------------------------------------- inference
    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        """Fast inference: single VLM forward → gather action tokens → MLP regress.
        NO langact decoding, NO denoise loop (OFT is one-shot).

        Optional ctxdemo prefix KV-cache reuse (`kv_cache_key=...`): the context
        demo is a bit-exact immutable prefix (~85% of the prompt tokens) for every
        step of an episode, so its transformer KV is computed once per episode and
        reused; only the per-step suffix (obs+wrist+state+task+🔍) is prefilled.
        The 🔍 action queries live in the suffix, so the OFT readout needs no
        prefix hidden states — pure KV reuse is mathematically identical to the
        full forward. First call runs BOTH paths and verifies (auto-fallback)."""
        if not isinstance(examples, list):
            examples = [examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        for ex in examples:
            imgs = ex.get("image", [])
            if not isinstance(imgs, list):
                imgs = [imgs]
            imgs = [to_pil_preserve(im) for im in imgs]
            if train_obs_image_size:
                imgs = resize_images(imgs, target_size=train_obs_image_size)
            ex["image"] = imgs

        device = next(self.action_model.parameters()).device
        qwen_inputs, _, _ = self._build_inference_inputs(examples)
        qwen_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in qwen_inputs.items()}

        # ── ctxdemo prefix KV-reuse fast path ──
        kv_key = kwargs.get("kv_cache_key")
        if (
            kv_key is not None
            and len(examples) == 1
            and examples[0].get("context_demo") is not None
            and not getattr(self, "_kv_disabled", False)
        ):
            try:
                pred_fast = self._predict_with_prefix_kv(examples, qwen_inputs, kv_key)
                if not getattr(self, "_kv_verified", False):
                    # One-time cross-check vs the full forward (per server process).
                    pred_slow = self._predict_full_forward(qwen_inputs)
                    diff = (pred_fast.float() - pred_slow.float()).abs().max().item()
                    logger.info(f"[kv_reuse] verification max|Δaction|={diff:.2e}")
                    if diff > 5e-2:
                        raise RuntimeError(f"verification failed (max diff {diff:.3e})")
                    self._kv_verified = True
                return {"normalized_actions": pred_fast.detach().cpu().numpy()}
            except Exception as e:  # noqa: BLE001 — any incompat → permanent fallback
                logger.warning(f"[kv_reuse] fast path failed ({e!r}); disabling, using full forward")
                self._kv_disabled = True
                self._kv_cache = None

        pred = self._predict_full_forward(qwen_inputs)
        return {"normalized_actions": pred.detach().cpu().numpy()}

    def _predict_full_forward(self, qwen_inputs: dict) -> torch.Tensor:
        """The original single full-prompt forward → 🔍 gather → MLP head."""
        out = self.qwen_vl_interface(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = out.hidden_states[-1]
        input_ids = qwen_inputs["input_ids"]
        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, self.action_token_id
            )
            pred = self.action_model.predict_action(action_queries)
        return pred

    # ------------------------------------------------------- kv-reuse internals
    def _kv_prefix_boundary(self, input_ids_row: torch.Tensor, n_demo_images: int) -> int:
        """Token index of the (n_demo_images+1)-th <|vision_start|> — i.e. the
        first token of the CURRENT-obs image block = end of the immutable ctxdemo
        prefix (demo header text + N demo images + demo motion summary + '---')."""
        vs_id = self.qwen_vl_interface.processor.tokenizer.convert_tokens_to_ids(
            "<|vision_start|>"
        )
        pos = (input_ids_row == vs_id).nonzero(as_tuple=False).flatten()
        if pos.numel() < n_demo_images + 1:
            return -1
        return int(pos[n_demo_images].item())

    def _predict_with_prefix_kv(
        self, examples: List[dict], qwen_inputs: dict, kv_key: str
    ) -> torch.Tensor:
        """Prefill the immutable demo prefix once per episode (cache its KV), then
        per step forward only the suffix with `past_key_values`. M-RoPE 3D
        positions are computed on the FULL sequence and sliced, so suffix
        positions are exactly those of the full forward."""
        from transformers.cache_utils import DynamicCache

        vlm_if = self.qwen_vl_interface
        input_ids = qwen_inputs["input_ids"]            # (1, L)
        grid = qwen_inputs["image_grid_thw"]            # (N_img, 3)
        pix = qwen_inputs["pixel_values"]               # (total_patches, D)
        mm_tt = qwen_inputs.get("mm_token_type_ids")    # (1, L) or None
        device = input_ids.device

        n_demo = len(examples[0]["context_demo"].get("demo_frames") or [])
        if n_demo <= 0:
            raise RuntimeError("no demo frames")
        P = self._kv_prefix_boundary(input_ids[0], n_demo)
        if P <= 0:
            raise RuntimeError("prefix boundary not found")

        # Full-sequence 3D M-RoPE positions (cheap; guarantees suffix positions
        # match the full forward bit-for-bit).
        inner = vlm_if.model
        rope_fn = getattr(inner, "get_rope_index", None) or getattr(
            inner.model, "get_rope_index"
        )
        full_pos, _ = rope_fn(
            input_ids, grid, attention_mask=torch.ones_like(input_ids)
        )                                               # (3, 1, L)

        demo_patches = int(grid[:n_demo].prod(dim=1).sum().item())

        cache = getattr(self, "_kv_cache", None)
        prefix_ids_cpu = input_ids[0, :P].cpu()
        if (
            cache is None
            or cache["key"] != kv_key
            or cache["P"] != P
            or not torch.equal(cache["prefix_ids"], prefix_ids_cpu)
        ):
            past = DynamicCache()
            pre_kwargs = dict(
                input_ids=input_ids[:, :P],
                attention_mask=torch.ones_like(input_ids[:, :P]),
                position_ids=full_pos[:, :, :P],
                pixel_values=pix[:demo_patches],
                image_grid_thw=grid[:n_demo],
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            if mm_tt is not None:
                pre_kwargs["mm_token_type_ids"] = mm_tt[:, :P]
            vlm_if(**pre_kwargs)
            cache = {"key": kv_key, "P": P, "prefix_ids": prefix_ids_cpu, "past": past}
            self._kv_cache = cache
        past = cache["past"]

        S = input_ids.shape[1] - P
        suf_kwargs = dict(
            input_ids=input_ids[:, P:],
            attention_mask=torch.ones(1, P + S, device=device, dtype=torch.long),
            position_ids=full_pos[:, :, P:],
            pixel_values=pix[demo_patches:],
            image_grid_thw=grid[n_demo:],
            past_key_values=past,
            cache_position=torch.arange(P, P + S, device=device),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        if mm_tt is not None:
            suf_kwargs["mm_token_type_ids"] = mm_tt[:, P:]
        out = vlm_if(**suf_kwargs)

        # The suffix forward appended S entries to the shared prefix cache — crop
        # them off so the cache stays the pristine per-episode prefix.
        if hasattr(past, "crop"):
            past.crop(P)
        else:  # very old transformers: rebuild next call
            self._kv_cache = None

        last_hidden = out.hidden_states[-1]             # (1, S, H)
        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids[:, P:], self.action_token_id
            )
            pred = self.action_model.predict_action(action_queries)
        return pred

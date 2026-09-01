# Copyright 2026 stellavla community.
# Licensed under the MIT License.
"""
StellaVLA policy server for closed-loop LIBERO evaluation.

StellaVLA was trained with `use_context_demo=True` and `include_wrist_image=True`,
so at inference the prompt must carry the same structure: demo keyframes, the
demo's task plan and sub-task text, the wrist view and the current view. Serving
without the demo puts the VLM features out of distribution and success collapses.

Per request this server attaches:
  - `context_demo`: this episode's demo, read from the released demo pack and
    locked per (suite, task_id, episode_id).
  - `image` = [primary, wrist] and `state`, both supplied by the client.

Prompt construction itself lives in the framework, so the server only attaches
the demo and forwards to `vla.predict_action`.

Usage:
    python examples/LIBERO/eval_files/serve_stellavla.py \
        --ckpt_path <run>/checkpoints/steps_30000_pytorch_model.pt \
        --port 6500 --use_context_demo --demo_pack <pack>/manifest.json \
        --max_subgoals 10 --demo_seed 42

Client payload (per example):
    {"lang": str, "image": [primary, wrist], "state": [8],
     "suite": str, "task_id": int, "episode_id": str}
"""

import argparse
import json
import logging
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from stellavla.model.framework.base_framework import baseframework

logging.basicConfig(level=logging.INFO, force=True)
log = logging.getLogger("serve_stellavla")


class StellaVLAPolicy:
    """Wrap a StellaVLA checkpoint for WebsocketPolicyServer, injecting the demo."""

    def __init__(self, ckpt_path: str,
                 use_context_demo: bool = True,
                 task_classification: Optional[str] = None,
                 max_subgoals: int = 10,
                 demo_seed: int = 42,
                 no_gripper_2d: bool = False,
                 kv_reuse: bool = False,
                 wrong_demo: bool = False,
                 demo_strip: str = "none",
                 demo_pack: Optional[str] = None):
        # from_pretrained reads config.yaml from the run dir above the .pt,
        # builds the framework through FRAMEWORK_REGISTRY, and loads the weights.
        # The VLM comes from config.framework.qwenvl.base_vlm.
        log.info(f"[stellavla] loading framework from {ckpt_path}")
        self.vla = baseframework.from_pretrained(ckpt_path)
        self.vla = self.vla.to("cuda").eval()
        log.info("[stellavla] framework loaded + on cuda")

        self.use_context_demo = bool(use_context_demo)
        # no_gripper_2d drops the "Steering Command 2D" ask from the prompt, for
        # a model trained with gripper_2d_dropout=1.0.
        self.no_gripper_2d = bool(no_gripper_2d)
        self.max_subgoals = int(max_subgoals)
        self._demo_seed = int(demo_seed)
        # kv_reuse: cache the demo-prefix transformer KV once per episode and
        # reuse it every step. The demo is a bit-exact immutable prefix and ~85%
        # of the prompt tokens. OFT head only; the first call is verified against
        # a full forward and falls back on any mismatch.
        self.kv_reuse = bool(kv_reuse)
        self._lat_ms: list = []
        # Ablations, applied to the payload only:
        #   wrong_demo: inject a different task's demo (the next one in the suite,
        #     deterministically) — a content-vs-format control.
        #   demo_strip: 'text' keeps only the images, 'image' keeps only the text.
        self.wrong_demo = bool(wrong_demo)
        assert demo_strip in ("none", "text", "image"), demo_strip
        self.demo_strip = demo_strip
        # Per-episode demo lock cache: (suite, task_id, episode_id) -> payload.
        self._demo_cache: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
        # Per-suite {normalized instruction -> lerobot task_index}. Needed because
        # the eval task_id (LIBERO benchmark order) != lerobot task_index, so the
        # demo must be resolved by INSTRUCTION TEXT, not task_id.
        self._lang2idx: Dict[str, Dict[str, int]] = {}
        # LIBERO-plus base-task-identity demo resolution: the Language perturbation
        # rewords instructions (heavy paraphrases) so text/prefix match fails; but
        # each perturbed task's NAME embeds the underscored base instruction, so we
        # recover the correct base task_index from it. Enabled by --task_classification
        # (LIBERO-plus task_classification.json); None => disabled (standard eval).
        self._task_classification_path = task_classification
        self._tc: Dict[str, list] = {}       # suite -> classification list (lazy)
        self._base_us: Dict[str, list] = {}  # suite -> [(base_underscored, idx)] longest-first (lazy)

        # The demo pack carries the resolved demo payloads plus the candidate
        # pool per task, so evaluation needs no trajectory or annotation corpus.
        self._demo_pack: Optional[Dict[str, Any]] = None
        self._demo_pack_root: Optional[Path] = None
        if not demo_pack and self.use_context_demo:
            raise ValueError("--use_context_demo requires --demo_pack")
        if demo_pack:
            mpath = Path(demo_pack)
            if mpath.is_dir():
                mpath = mpath / "manifest.json"
            self._demo_pack_root = mpath.parent
            with open(mpath) as f:
                self._demo_pack = json.load(f)
            self._lang2idx.update({k: dict(v) for k, v in
                                   self._demo_pack.get("lang_to_task", {}).items()})
            log.info(f"[ctxdemo] demo pack {mpath} "
                     f"(payloads={len(self._demo_pack.get('payloads', {}))} "
                     f"pools={len(self._demo_pack.get('pools', {}))} "
                     f"meta={self._demo_pack.get('meta', {}).get('version')})")

    def _load_task_classification(self, suite: str):
        """Lazy-load task_classification.json[suite] (list indexed by client task_id).
        Returns None when --task_classification not set (non-LIBERO-plus eval)."""
        if not self._task_classification_path:
            return None
        if suite not in self._tc:
            import json
            try:
                with open(self._task_classification_path) as f:
                    self._tc[suite] = json.load(f).get(suite, [])
            except Exception as e:
                log.warning(f"[ctxdemo] load task_classification failed ({suite}): {e}")
                self._tc[suite] = []
        return self._tc[suite]

    def _base_underscored(self, suite: str):
        """[(base_instruction_underscored, task_index)] for `suite`, longest-first,
        derived from self._lang2idx — for substring-matching a perturbed task's NAME
        (which embeds the underscored base instruction)."""
        if suite not in self._base_us:
            m = self._lang2idx.get(suite, {})
            self._base_us[suite] = sorted(
                ((k.replace(" ", "_"), v) for k, v in m.items()),
                key=lambda kv: -len(kv[0]),
            )
        return self._base_us[suite]

    def _resolve_demo_task_idx(self, suite: str, lang: Optional[str], fallback_task_id: int) -> int:
        """Map the eval instruction (lang) -> the lerobot task_index whose
        description matches. CRITICAL: eval task_id (LIBERO benchmark order) does
        NOT equal lerobot task_index, so using task_id injects the WRONG task's
        demo (e.g. spatial eval-t1 'next to ramekin' -> lerobot-t1 'top drawer').
        Match by instruction text instead. Falls back to task_id on miss."""
        if not lang:
            return int(fallback_task_id)
        s = str(suite)
        norm = " ".join(str(lang).strip().lower().split())
        idx = self._lang2idx.get(s, {}).get(norm)
        if idx is not None:
            return int(idx)
        # LIBERO-plus appends a variant suffix to the instruction (e.g. base
        # "...place it on the plate" -> "...place it on the plate table 1").
        # Resolve the base demo by longest base-instruction prefix (word-boundary
        # safe). Reworded Language-perturbation tasks won't prefix-match -> task_id.
        cands = [(k, v) for k, v in self._lang2idx.get(s, {}).items()
                 if norm.startswith(k) and (len(norm) == len(k) or norm[len(k)] == " ")]
        if cands:
            k, v = max(cands, key=lambda kv: len(kv[0]))
            return int(v)
        # Base-task-identity resolution (LIBERO-plus reworded Language tasks): the
        # perturbed task's NAME embeds the underscored base instruction, so match the
        # longest base instruction that is a substring of name[fallback_task_id].
        # Recovers the correct same-task demo for heavy paraphrases text-match misses.
        tc = self._load_task_classification(s)
        if tc and 0 <= int(fallback_task_id) < len(tc):
            name = str(tc[int(fallback_task_id)].get("name", "")).lower()
            for b_us, v in self._base_underscored(s):
                if b_us and b_us in name:
                    return int(v)
        log.warning(f"[ctxdemo] lang->task_idx miss ({s}): {str(lang)[:60]!r}; "
                    f"falling back to task_id={fallback_task_id}")
        return int(fallback_task_id)

    # ── demo retrieval (mirrors the high-level server demo retrieval) ──
    def _apply_demo_strip(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Modality ablation, shared by the corpus and the pack paths."""
        if self.demo_strip == "text":
            # Images-only demo: keep task line + subgoal frames; blank all guidance text.
            return dict(payload, demo_task_plan="", demo_initial_state="",
                        demo_final_state="", demo_frame_states=[],
                        demo_subtask_summary="")
        if self.demo_strip == "image":
            # Text-only demo: drop the subgoal frames (per-frame state captions
            # drop with them — they pair 1:1 with images in the prompt builder).
            return dict(payload, demo_frames=[], demo_frame_states=[])
        return payload

    def _pack_payload(self, suite: str, demo_task_idx: int,
                      episode_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Resolve this episode's demo from the pack.

        The pick is a pure function of (demo_seed, suite, demo_task_idx,
        episode_id), so it does not depend on how the episodes were sharded
        across workers or GPUs and the score reproduces across machines.
        """
        import hashlib
        from PIL import Image
        pool = (self._demo_pack.get("pools") or {}).get(f"{suite}|t{int(demo_task_idx)}") or []
        if not pool:
            return None
        ident = f"{self._demo_seed}|{suite}|{int(demo_task_idx)}|{episode_id}"
        seed = int(hashlib.sha1(ident.encode()).hexdigest()[:12], 16)
        ep = pool[int(np.random.default_rng([seed, 0x4453]).integers(len(pool)))]
        pid = f"{suite}#{int(ep)}"
        entry = (self._demo_pack.get("payloads") or {}).get(pid)
        if entry is None:
            log.warning(f"[ctxdemo] pack miss for {pid} (task_idx={demo_task_idx})")
            return None
        try:
            frames = [Image.open(self._demo_pack_root / rel).convert("RGB")
                      for rel in entry.get("frames", [])]
        except Exception as e:
            log.warning(f"[ctxdemo] pack frame load failed for {pid}: {e}")
            return None
        return {
            "demo_instruction": entry.get("demo_instruction", ""),
            "demo_frames": frames,
            "demo_frame_states": entry.get("demo_frame_states", []),
            "demo_initial_state": entry.get("demo_initial_state", ""),
            "demo_final_state": entry.get("demo_final_state", ""),
            "demo_task_plan": entry.get("demo_task_plan", ""),
            "demo_subtask_summary": entry.get("demo_subtask_summary", ""),
        }

    def _get_demo_payload(self, suite: str, task_id: int,
                          episode_id: Optional[str], lang: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self._demo_pack is None:
            return None
        key: Optional[Tuple[str, int, str]] = None
        if episode_id is not None:
            key = (str(suite), int(task_id), str(episode_id))
            cached = self._demo_cache.get(key)
            if cached is not None:
                return cached
        # Resolve the demo's lerobot task_index by INSTRUCTION TEXT (not task_id).
        demo_task_idx = self._resolve_demo_task_idx(suite, lang, task_id)
        if self.wrong_demo:
            # Ablation: deterministically inject the NEXT task's demo instead —
            # a same-suite but WRONG-task demonstration (content-vs-format control).
            pref = f"{suite}|t"
            tasks = sorted(int(k[len(pref):]) for k in
                           (self._demo_pack.get("pools") or {}) if k.startswith(pref))
            if tasks:
                try:
                    wrong = tasks[(tasks.index(int(demo_task_idx)) + 1) % len(tasks)]
                except ValueError:
                    wrong = tasks[0]
                log.info(f"[wrong_demo] task_idx {demo_task_idx} -> {wrong} ({suite})")
                demo_task_idx = wrong
        payload = self._pack_payload(str(suite), int(demo_task_idx), episode_id)
        if payload is None:
            return None
        payload = self._apply_demo_strip(payload)
        if key is not None:
            self._demo_cache[key] = payload
        return payload

    @torch.inference_mode()
    def predict_action(self, examples=None, use_ddim: bool = False,
                       num_ddim_steps: Optional[int] = None, **kwargs) -> dict:
        """Attach context_demo per example, then forward to the framework.

        Client sends each example as:
          {"lang", "image": [primary, wrist], "state": [8],
           "suite", "task_id", "episode_id"}

        """
        if examples is None:
            raise ValueError("predict_action: examples is None")
        kv_cache_key = None
        for ex in examples:
            if self.no_gripper_2d:
                ex["has_gripper_2d"] = False
            if self.use_context_demo:
                suite = ex.get("suite")
                tid = ex.get("task_id")
                epid = ex.get("episode_id")
                if suite is not None and tid is not None:
                    demo = self._get_demo_payload(suite, int(tid), epid, lang=ex.get("lang"))
                    if demo is not None and (demo.get("demo_frames") or demo.get("demo_subtask_summary")):
                        ex["context_demo"] = demo
                        if self.kv_reuse and len(examples) == 1 and demo.get("demo_frames"):
                            # Per-episode key: the demo payload (hence the token
                            # prefix) is locked per (suite, task_id, episode_id).
                            # (No frames → no image-anchored prefix boundary → skip KV.)
                            # Safe because the demo for an episode is fixed for
                            # the life of the process.
                            kv_cache_key = f"{suite}|{tid}|{epid}"
                    else:
                        log.warning(f"[ctxdemo] demo retrieval failed suite={suite} "
                                    f"task_id={tid} ep={epid}; running WITHOUT demo (OOD)")
        _t0 = time.perf_counter()
        out = self.vla.predict_action(
            examples=examples, use_ddim=use_ddim, num_ddim_steps=num_ddim_steps,
            kv_cache_key=kv_cache_key,
        )
        self._lat_ms.append((time.perf_counter() - _t0) * 1e3)
        if len(self._lat_ms) % 25 == 0:
            lat = sorted(self._lat_ms[-100:])
            log.info(f"[latency] predict_action n={len(self._lat_ms)} "
                     f"kv_reuse={self.kv_reuse} last100: "
                     f"mean={sum(lat)/len(lat):.0f}ms p50={lat[len(lat)//2]:.0f}ms "
                     f"min={lat[0]:.0f}ms max={lat[-1]:.0f}ms")
        return out


def main(args):
    policy = StellaVLAPolicy(
        ckpt_path=args.ckpt_path,
        use_context_demo=args.use_context_demo,
        demo_pack=args.demo_pack,
        task_classification=args.task_classification,
        max_subgoals=args.max_subgoals,
        demo_seed=args.demo_seed,
        no_gripper_2d=args.no_gripper_2d,
        kv_reuse=args.kv_reuse,
        wrong_demo=args.wrong_demo,
        demo_strip=args.demo_strip,
    )
    hostname = socket.gethostname()
    log.info(f"[stellavla] server binding 0.0.0.0:{args.port} on {hostname}")
    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"server_kind": "stellavla_hf", "host": hostname},
    )
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Checkpoint .pt; config.yaml must sit in the run dir above checkpoints/")
    parser.add_argument("--port", type=int, default=6500)
    parser.add_argument("--idle_timeout", type=int, default=-1)
    parser.add_argument("--use_context_demo", action="store_true", default=False)
    parser.add_argument("--demo_pack", type=str, default=None,
                        help="Released demo pack (directory or manifest.json)")
    parser.add_argument("--task_classification", type=str, default=None,
                        help="LIBERO-plus task_classification.json; enables base-task-identity "
                             "demo resolution for reworded (Language-perturbation) instructions.")
    parser.add_argument("--no_gripper_2d", action="store_true", default=False,
                        help="Force has_gripper_2d=False on every example (eval the no2d model "
                             "trained with gripper_2d_dropout=1.0 — prompt omits the 2D ask).")
    parser.add_argument("--max_subgoals", type=int, default=10)
    parser.add_argument("--demo_seed", type=int, default=42)
    parser.add_argument("--kv_reuse", action="store_true", default=False,
                        help="Cache the ctxdemo-prefix transformer KV once per episode and "
                             "reuse it every step (demo ≈85%% of prompt tokens → ~5-7x faster "
                             "VLM prefill). OFT-head only; first call verifies vs the full "
                             "forward and auto-falls-back on mismatch.")
    parser.add_argument("--wrong_demo", action="store_true", default=False,
                        help="ABLATION: inject the NEXT task's demo instead of the matching "
                             "one (deterministic wrong-task control — tests whether the model "
                             "uses demo CONTENT, not just its format).")
    parser.add_argument("--demo_strip", type=str, default="none",
                        choices=["none", "text", "image"],
                        help="ABLATION: strip the demo payload — 'text' = images-only demo "
                             "(blank plan/states/motions), 'image' = text-only demo (no "
                             "subgoal frames).")
    args = parser.parse_args()
    main(args)

# Copyright 2026 stellavla community.
# Licensed under the MIT License.
"""
StellaVLA policy server for closed-loop VLA-Arena evaluation.

Same framework and prompt scaffold as the LIBERO server; the VLA-Arena deltas are:

  * DEMO: resolved from the released demo pack, keyed by
    `<suite>|L<level>|t<task_id>` (falling back to the normalized instruction).
    A cell with no demo is a genuine no-demo task, not a lookup miss.
  * STATE TEXT: VLA-Arena carries a single gripper qpos at `state[7]`
    (`state[6]` is an unused pad), so the "Current robot state:" line is
    rendered with the index-7 variant to match training.
  * NO-DEMO ABLATION (`--no_demo`): keep `use_context_demo=True` on the
    framework so the prompt scaffold still matches training, but never attach a
    demo payload.
  * norm_stats / chunk_len are exposed so the generic server un-normalizes the
    action chunk (min/max, binary {0,1} gripper) and the client learns the
    chunk size.

Usage:
    python examples/VLA-Arena/eval_files/serve_stellavla.py \
        --ckpt_path <run>/checkpoints/steps_30000_pytorch_model.pt \
        --port 10090 --use_context_demo --demo_pack <pack>/manifest.json \
        --max_subgoals 10

Client payload (per example):
    {"lang": str, "image": [primary, wrist], "state": [8],
     "suite": str, "level": int, "task_id": int}
"""

import argparse
import json
import logging
import math
import socket
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from stellavla.model.framework.base_framework import baseframework

logging.basicConfig(level=logging.INFO, force=True)
log = logging.getLogger("serve_vla_arena_stellavla")


# VLA-Arena gripper: a single qpos at state[7] over the [closed, open] finger
# range. abs() keeps it sign-robust whether finger2 is the +[0, 0.04] joint or
# its mirror.
_GRIP_QPOS_CLOSED = 0.0
_GRIP_QPOS_OPEN = 0.04


def _describe_robot_state(state_vec) -> str:
    """VLA-Arena 8-dim end-effector state -> the language the model was trained on."""
    if state_vec is None or len(state_vec) < 6:
        return ""
    x, y, z = (int(round(float(state_vec[i]) * 1000)) for i in range(3))                # m -> mm
    r, pi, yw = (int(round(float(state_vec[i]) * 180.0 / math.pi)) for i in range(3, 6))  # rad -> deg
    if state_vec is None or len(state_vec) < 8:
        g = 0
    else:
        frac = (abs(float(state_vec[7])) - _GRIP_QPOS_CLOSED) / (_GRIP_QPOS_OPEN - _GRIP_QPOS_CLOSED)
        g = int(round(min(1.0, max(0.0, frac)) * 100))
    return (f"end-effector position x={x} y={y} z={z} mm; "
            f"orientation roll={r} pitch={pi} yaw={yw} deg; gripper open {g}%")


def _norm_lang(lang: Optional[str]) -> str:
    """Normalize an instruction identically to the LIBERO serve's
    `_resolve_demo_task_idx` (lowercase, collapse interior whitespace)."""
    return " ".join(str(lang).strip().lower().split()) if lang else ""


class StellaVLAPolicy:
    """Wrap a StellaVLA checkpoint for WebsocketPolicyServer, injecting the demo."""

    def __init__(self, ckpt_path: str,
                 use_context_demo: bool = True,
                 no_demo: bool = False,
                 max_subgoals: int = 10,
                 demo_pack: Optional[str] = None):
        # from_pretrained reads config.yaml from the run dir above the .pt,
        # builds the framework through FRAMEWORK_REGISTRY and loads the weights.
        log.info(f"[stellavla] loading framework from {ckpt_path}")
        self.vla = baseframework.from_pretrained(ckpt_path)
        self.vla = self.vla.to("cuda").eval()
        log.info("[stellavla] framework loaded + on cuda")

        # The generic server un-normalizes actions by reading these off the policy.
        self.norm_stats = getattr(self.vla, "norm_stats", None)
        self.chunk_len = int(getattr(self.vla, "chunk_len", 8))
        log.info(f"[stellavla] norm_stats keys="
                 f"{list(self.norm_stats.keys()) if self.norm_stats else None} "
                 f"chunk_len={self.chunk_len}")

        self.use_context_demo = bool(use_context_demo)
        self._no_demo = bool(no_demo)
        self.max_subgoals = int(max_subgoals)
        # The demo for a cell is fixed, so cache the built payload per demo.
        self._demo_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}

        self._demo_pack: Optional[Dict[str, Any]] = None
        self._demo_pack_root: Optional[Path] = None
        if not demo_pack and self.use_context_demo and not self._no_demo:
            raise ValueError("--use_context_demo requires --demo_pack")
        if demo_pack:
            mpath = Path(demo_pack)
            if mpath.is_dir():
                mpath = mpath / "manifest.json"
            self._demo_pack_root = mpath.parent
            with open(mpath) as f:
                self._demo_pack = json.load(f)
            _meta = self._demo_pack.get("meta", {})
            log.info(f"[ctxdemo] demo pack loaded from {mpath} "
                     f"(cells={len(self._demo_pack.get('cells', {}))} meta={_meta})")

    # ── demo retrieval ────────────────────────────────────────────────────
    def _pack_payload(self, suite, level, task_id, lang) -> Optional[Dict[str, Any]]:
        """Rebuild the demo payload from the pack.

        The pack stores the six text fields and the subgoal frames as lossless
        PNG, so what reaches the model is what the payload builder produced.
        """
        from PIL import Image
        cells = self._demo_pack.get("cells", {})
        cell = None
        if suite is not None and level is not None and task_id is not None:
            cell = cells.get(f"{suite}|L{int(level)}|t{int(task_id)}")
        if cell is None and lang:
            _n = _norm_lang(lang)
            for c in cells.values():
                if c.get("instruction_norm") == _n:
                    cell = c
                    break
        if cell is None:
            return None
        demos = cell.get("demos") or []
        if not demos:
            return None                      # a genuine no-demo task
        d = demos[0]
        ckey = (str(d.get("demo_unit")), int(d.get("demo_ep_idx", -1)))
        cached = self._demo_cache.get(ckey)
        if cached is not None:
            return cached
        try:
            frames = [Image.open(self._demo_pack_root / rel).convert("RGB")
                      for rel in d.get("frames", [])]
        except Exception as e:
            log.warning(f"[ctxdemo] pack frame load failed for {cell.get('suite')}: {e}")
            return None
        payload = {
            "demo_instruction": d.get("demo_instruction", ""),
            "demo_frames": frames,
            "demo_frame_states": d.get("demo_frame_states", []),
            "demo_initial_state": d.get("demo_initial_state", ""),
            "demo_final_state": d.get("demo_final_state", ""),
            "demo_task_plan": d.get("demo_task_plan", ""),
            "demo_subtask_summary": d.get("demo_subtask_summary", ""),
        }
        self._demo_cache[ckey] = payload
        return payload

    def _get_demo_payload(self, suite: Optional[str], level: Optional[Any],
                          task_id: Optional[Any], lang: Optional[str]) -> Optional[Dict[str, Any]]:
        # --no_demo ablation: keep the prompt scaffold (use_context_demo=True on
        # the framework) but never attach a demo, so predict_action runs the
        # no-demo branch for EVERY request.
        if self._no_demo:
            return None
        if self._demo_pack is None:
            return None
        return self._pack_payload(suite, level, task_id, lang)

    @torch.inference_mode()
    def predict_action(self, examples=None, use_ddim: bool = False,
                       num_ddim_steps: Optional[int] = None, **kwargs) -> dict:
        """Attach context_demo and state_text per example, then forward to the
        framework.

        Client sends each example as:
          {"lang", "image": [primary, wrist], "state": [8],
           "suite", "level", "task_id"}

        """
        if examples is None:
            raise ValueError("predict_action: examples is None")
        for ex in examples:
            # state_text: render the client's 8-dim VLA-Arena eef state (index-7
            # gripper) so the "Current robot state:" prompt line matches training.
            if ex.get("state") is not None and not ex.get("state_text"):
                try:
                    ex["state_text"] = _describe_robot_state(ex["state"])
                except Exception as e:
                    log.warning(f"[state_text] render failed: {e}")
            if self.use_context_demo:
                suite = ex.get("suite")
                level = ex.get("level")
                tid = ex.get("task_id")
                lang = ex.get("lang")
                demo = self._get_demo_payload(suite, level, tid, lang)
                if demo is not None and demo.get("demo_frames"):
                    ex["context_demo"] = demo
                elif not self._no_demo:
                    log.warning(f"[ctxdemo] demo retrieval failed suite={suite} "
                                f"level={level} task_id={tid} lang={str(lang)[:40]!r}; "
                                f"running WITHOUT demo (OOD)")
        out = self.vla.predict_action(
            examples=examples, use_ddim=use_ddim, num_ddim_steps=num_ddim_steps,
        )
        return out


def main(args):
    policy = StellaVLAPolicy(
        ckpt_path=args.ckpt_path,
        use_context_demo=args.use_context_demo,
        demo_pack=args.demo_pack,
        no_demo=args.no_demo,
        max_subgoals=args.max_subgoals,
    )
    hostname = socket.gethostname()
    log.info(f"[stellavla] server binding 0.0.0.0:{args.port} on {hostname}")
    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={
            "server_kind": "stellavla_vla_arena_hf",
            "env": "simpler_env",
            "action_chunk_size": policy.chunk_len,  # client REQUIRES this key
            "gripper_passthrough": False,            # binary {0,1} gripper (server binarizes dim 6)
            "norm_mode": args.action_norm_mode,      # ④ un-norm mode; MUST match training
            "host": hostname,
        },
    )
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Checkpoint .pt; config.yaml must sit in the run dir above checkpoints/")
    parser.add_argument("--port", type=int, default=10090)
    parser.add_argument("--idle_timeout", type=int, default=-1)
    parser.add_argument("--use_context_demo", action="store_true", default=False)
    parser.add_argument("--demo_pack", type=str, default=None,
                        help="Released demo pack (directory or manifest.json)")
    parser.add_argument("--no_demo", action="store_true", default=False,
                        help="ABLATION: keep the use_context_demo prompt scaffold but never "
                             "attach a demo payload.")
    parser.add_argument("--max_subgoals", type=int, default=10)
    # ④ MUST match the model's training-time action_norm_mode. min_max (default,
    # StellaVLA) un-normalizes with stats min/max; q99 uses q01/q99. A mismatch
    # silently degrades actions (VLA-Arena 82% -> 20%).
    parser.add_argument("--action_norm_mode", type=str, default="min_max",
                        choices=["min_max", "q99"])
    args = parser.parse_args()
    main(args)

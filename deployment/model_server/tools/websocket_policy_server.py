# Copyright 2025 stellavla community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import logging
import traceback
import time

import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # seconds; -1 = never shut down
        metadata: dict | None = None,
        
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        # Disable keepalive ping — long-running model.generate() inside the
        # handler blocks the asyncio event loop, which means outgoing pings
        # never receive a pong before the default 20 s ping_timeout fires
        # (1011 internal error). Client (websocket_policy_client.py) sets
        # ping_interval=60/timeout=120, but the SERVER's own ping is what
        # was killing connections under HL load (~25-30 s per inference,
        # × N concurrent workers serialized = far over 20 s). Setting
        # ping_interval=None turns off the server-initiated keepalive
        # entirely — the client's own keepalive is enough to detect dead
        # connections. Symptom this fixed: 4-worker LIBERO sim eval where
        # libero_object/libero_10 always 0/100 due to every HL call dying
        # with "received 1011 ... keepalive ping timeout".
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
        ) as server:
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server):
        """Shut the server down once it has been idle for `idle_timeout` seconds."""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()
                ret = self._route_message(msg)  # route message
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    # route logic: recognize request from client
    def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")          # default = infer
        msg       # when no explicit payload, treat top-level as payload

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # infer --> framework.predict_action
        elif mtype == "infer" or mtype == "predict_action":
            # Basic payload sanity
            if not isinstance(msg, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(msg))}
                }
            try:

                ouput_dict = self._policy.predict_action(**msg)
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                logging.exception(e)
                
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                        # "traceback": traceback.format_exc(),
                    },
                }
            data = ouput_dict
            # Un-normalize actions server-side so clients receive ready-to-use
            # actions under data["actions"]. Frameworks' predict_action returns
            # only "normalized_actions"; clients (eval interfaces) lack the norm
            # stats, so the un-normalization must happen here. Guarded + additive
            # (keeps "normalized_actions"), so it is backward-compatible.
            if (
                isinstance(data, dict)
                and "normalized_actions" in data
                and "actions" not in data
                and getattr(self._policy, "norm_stats", None)
            ):
                try:
                    import numpy as _np

                    # Read the INSTANCE norm_stats (set in from_pretrained); the
                    # framework's get_action_stats is a classmethod and would look
                    # the attribute up on the class instead.
                    _ns = self._policy.norm_stats
                    _uk = msg.get("unnorm_key", None) or next(iter(_ns.keys()))
                    _stats = _ns[_uk]["action"]
                    # Un-normalize with the SAME mode used in training, selected via
                    # server metadata "norm_mode" (default min_max = StellaVLA). A
                    # mismatch silently degrades actions (e.g. VLA-Arena 82% -> 20%),
                    # which is exactly why base_framework's unconditional q01/q99
                    # unnormalize_actions is bypassed here. Set norm_mode="q99" in the
                    # serve script when the model was trained with action_norm_mode=q99.
                    _norm_mode = self._metadata.get("norm_mode", "min_max")
                    if _norm_mode == "q99":
                        _low = _np.array(_stats["q01"])
                        _high = _np.array(_stats["q99"])
                    else:
                        _low = _np.array(_stats["min"])
                        _high = _np.array(_stats["max"])
                    _mask = _np.array(_stats.get("mask", _np.ones_like(_low, dtype=bool)))
                    # Gripper (action dim 6): LIBERO and VLA-Arena take a binary
                    # {0,1} gripper (0 = open), while some environments take a
                    # continuous -1 = open / +1 = close command. The model is
                    # trained on the raw bimodal value (mask[6]=False, never
                    # normalized), so binarizing would map a -1 = open command to
                    # 0 (half-open) and break grasping. Deployments that need the
                    # continuous form set gripper_passthrough.
                    _binarize_gripper = not self._metadata.get("gripper_passthrough", False)

                    def _unnorm_one(_a):
                        _a = _np.clip(_a, -1, 1)
                        if _binarize_gripper and _a.shape[-1] > 6:
                            _a[..., 6] = _np.where(_a[..., 6] < 0.5, 0, 1)
                        return _np.where(_mask, 0.5 * (_a + 1) * (_high - _low) + _low, _a)

                    _na = _np.asarray(data["normalized_actions"])
                    if _na.ndim == 3:
                        data["actions"] = _np.stack([_unnorm_one(_na[b].copy()) for b in range(_na.shape[0])])
                    else:
                        data["actions"] = _unnorm_one(_na.copy())
                except Exception:
                    logging.exception("Action un-normalization failed (request_id=%s)", req_id)
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": data,
            }

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
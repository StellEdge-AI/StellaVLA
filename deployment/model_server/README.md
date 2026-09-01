# Policy server

`tools/websocket_policy_server.py` is the transport both benchmarks serve over:
a WebSocket server that takes a batch of examples, calls the policy's
`predict_action`, un-normalizes the action chunk using the policy's
`norm_stats`, and returns it msgpack-encoded.

The benchmark-specific servers wrap a StellaVLA checkpoint in a policy object
and hand it to this server:

* `examples/LIBERO/eval_files/serve_stellavla.py` — LIBERO and LIBERO-plus
* `examples/VLA-Arena/eval_files/serve_stellavla.py` — VLA-Arena

`tools/websocket_policy_client.py` is the matching client, used by the
`model2*_interface.py` adapters on the simulator side.

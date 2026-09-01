"""VLM backbone factory.

StellaVLA is built on Qwen3-VL. `framework.qwenvl.vlm_family` is the explicit
selector — needed because `base_vlm` often points at a local fine-tuned ckpt
whose directory name carries no family hint.
"""


def get_vlm_model(config):
    vlm_name = str(config.framework.qwenvl.base_vlm)

    try:
        family = str(config.framework.qwenvl.get("vlm_family", "")).strip().lower() or None
    except Exception:
        family = None

    if family == "qwen3vl" or "Qwen3-VL" in vlm_name:
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)

    raise NotImplementedError(
        f"VLM backbone '{vlm_name}' is not supported (family={family!r}). "
        f"StellaVLA expects Qwen3-VL; set `framework.qwenvl.vlm_family: qwen3vl` "
        f"when initializing from a fine-tuned checkpoint whose path does not "
        f"contain 'Qwen3-VL'."
    )

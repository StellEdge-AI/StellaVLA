"""
Shared configuration / utility helpers for framework components:
- NamespaceWithGet: lightweight namespace behaving like a dict
- OmegaConf conversion helpers
- Config merging decorator for model __init__
- Checkpoint config/statistics loading
"""

import functools
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from omegaconf import OmegaConf


from stellavla.utils import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class NamespaceWithGet(SimpleNamespace):
    def get(self, key, default=None):
        """
        Return attribute value if present, else default (dict-like API).

        Args:
            key: Attribute name.
            default: Fallback if attribute missing.

        Returns:
            Any: Stored value or default.
        """
        return getattr(self, key, default)

    def items(self):
        """
        Iterate (key, value) pairs like dict.items().

        Returns:
            Generator[Tuple[str, Any], None, None]
        """
        return ((key, getattr(self, key)) for key in self.__dict__)

    def __iter__(self):
        """
        Return iterator over attribute keys (enables dict unpacking **obj).

        Returns:
            Iterator[str]
        """
        return iter(self.__dict__)

    def to_dict(self):
        """
        Recursively convert nested NamespaceWithGet objects into plain dicts.

        Returns:
            dict: Fully materialized dictionary structure.
        """
        return {key: value.to_dict() if isinstance(value, NamespaceWithGet) else value for key, value in self.items()}


def dict_to_namespace(d):
    """
    Create an OmegaConf config from a plain dictionary.

    Args:
        d: Input dictionary.

    Returns:
        OmegaConf: DictConfig instance.
    """
    return OmegaConf.create(d)


def _to_omegaconf(x: Any):
    """
    Convert diverse input types into an OmegaConf object.

    Accepted types:
        - None -> empty DictConfig
        - str path -> load YAML/JSON via OmegaConf.load
        - dict -> DictConfig
        - DictConfig / ListConfig -> returned unchanged
        - NamespaceWithGet / SimpleNamespace -> converted via vars()/to_dict()

    Args:
        x: Input candidate.

    Returns:
        OmegaConf: Normalized configuration node.
    """
    if x is None:
        return OmegaConf.create({})
    if isinstance(x, OmegaConf.__class__):  # fallback, typically not hit
        return x
    try:
        # OmegaConf node detection
        from omegaconf import DictConfig, ListConfig

        if isinstance(x, (DictConfig, ListConfig)):
            return x
    except Exception:
        pass

    if isinstance(x, str):
        # treat as path
        return OmegaConf.load(x)
    if isinstance(x, dict):
        return OmegaConf.create(x)
    if isinstance(x, NamespaceWithGet) or isinstance(x, SimpleNamespace):
        # convert to plain dict
        try:
            d = x.to_dict() if hasattr(x, "to_dict") else vars(x)
        except Exception:
            d = vars(x)
        return OmegaConf.create(d)
    # fallback: try to create
    return OmegaConf.create(x)


def merge_pram_config(init):
    """
    Decorator for __init__ to unify config handling.

    Behavior:
        1. Extract 'config' kwarg / arg (path | dict | OmegaConf | namespace)
        2. Convert to OmegaConf
        3. Merge with explicitly passed init parameters (explicit overrides file)
        4. Attach merged config to self.config
        5. Call original __init__ with merged config

    Args:
        init: Original __init__ function.

    Returns:
        Wrapped initializer.
    """

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        # Map positional args to parameter names (excluding self)
        sig = inspect.signature(init)
        param_names = [name for i, (name, p) in enumerate(sig.parameters.items()) if i > 0]

        init_kwargs = {}
        for name, val in zip(param_names, args):
            init_kwargs[name] = val
        # override with explicit kwargs
        init_kwargs.update(kwargs)

        # get provided config (if any)
        provided_config = init_kwargs.get("config", None)

        loaded_cfg = _to_omegaconf(provided_config)

        # build params cfg from explicit init args (other than config)
        params = {k: v for k, v in init_kwargs.items() if k != "config"}
        params_cfg = OmegaConf.create(params) if params else OmegaConf.create({})

        # merge: loaded_cfg <- params_cfg (params override file)
        merged = OmegaConf.merge(loaded_cfg, params_cfg)

        # set on instance
        try:
            # prefer attaching OmegaConf directly
            self.config = merged
        except Exception:
            # fallback to dict
            self.config = OmegaConf.to_container(merged, resolve=True)

        # prepare kwargs for original init: ensure config is the merged OmegaConf
        call_kwargs = dict(init_kwargs)
        call_kwargs["config"] = merged

        # call original __init__ using keyword args only (safer)
        return init(self, **call_kwargs)

    return wrapper


def _resolve_dataset_statistics(run_dir: Path) -> Path:
    """Locate dataset_statistics.json with cross-node fallbacks.

    Order:
      1. <run_dir>/dataset_statistics.json
      2. $STELLAVLA_DATASET_STATISTICS env var
      3. <run_dir.parent>/dataset_statistics.json (group-level shared file)
      4. First sibling experiment under <run_dir.parent> that has one
    Returns the primary (run_dir) path if nothing else found, so the caller's
    assertion fires with a familiar message.
    """
    primary = run_dir / "dataset_statistics.json"
    if primary.exists():
        return primary

    override = os.environ.get("STELLAVLA_DATASET_STATISTICS")
    if override:
        p = Path(override)
        if p.exists():
            overwatch.info(f"Using STELLAVLA_DATASET_STATISTICS override `{p}`")
            return p

    parent_shared = run_dir.parent / "dataset_statistics.json"
    if parent_shared.exists():
        overwatch.info(f"Using shared dataset_statistics.json from parent dir `{parent_shared}`")
        return parent_shared

    if run_dir.parent.exists():
        for sib in sorted(run_dir.parent.glob("*/dataset_statistics.json")):
            overwatch.warning(
                f"`dataset_statistics.json` missing under {run_dir}; "
                f"falling back to sibling `{sib}` (set STELLAVLA_DATASET_STATISTICS to override)"
            )
            return sib

    return primary


def _rewrite_base_vlm(global_cfg: dict) -> None:
    """In-place rewrite framework.qwenvl.base_vlm to a path that exists locally.

    Order:
      1. $STELLAVLA_BASE_VLM_OVERRIDE env var (always wins)
      2. Original path if it already exists
      3. ./playground/Pretrained_models/<basename> if it exists (cwd-relative)
      4. Leave original untouched (HF hub will try to resolve it)
    """
    fw = global_cfg.get("framework") if isinstance(global_cfg, dict) else None
    if not isinstance(fw, dict):
        return
    qwenvl = fw.get("qwenvl")
    if not isinstance(qwenvl, dict):
        return
    orig = qwenvl.get("base_vlm")
    if not isinstance(orig, str) or not orig:
        return

    override = os.environ.get("STELLAVLA_BASE_VLM_OVERRIDE")
    if override:
        if override != orig:
            overwatch.info(f"Rewriting base_vlm: `{orig}` → `{override}` (STELLAVLA_BASE_VLM_OVERRIDE)")
        qwenvl["base_vlm"] = override
        return

    if os.path.exists(orig):
        return

    basename = os.path.basename(orig.rstrip("/"))
    if not basename:
        return
    for cand in (
        Path("playground/Pretrained_models") / basename,
        Path.cwd() / "playground" / "Pretrained_models" / basename,
    ):
        if cand.exists():
            resolved = str(cand.resolve())
            overwatch.info(f"Rewriting base_vlm: `{orig}` → `{resolved}` (auto-resolved local repo)")
            qwenvl["base_vlm"] = resolved
            return


def read_model_config(pretrained_checkpoint):
    """
    Load global model configuration and dataset normalization statistics
    associated with a saved checkpoint (.pt).

    Expected directory layout:
        <run_dir>/checkpoints/<name>.pt
        <run_dir>/config.json
        <run_dir>/dataset_statistics.json

    Args:
        pretrained_checkpoint: Path to a .pt checkpoint file.

    Returns:
        tuple:
            global_cfg (dict): Loaded config.json contents.
            norm_stats (dict): Dataset statistics for (de)normalization.

    Raises:
        FileNotFoundError: If checkpoint or required JSON files are missing.
        AssertionError: If file suffix or structure invalid.
    """
    if os.path.isfile(pretrained_checkpoint):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(pretrained_checkpoint))}`")

        # [Validate] Checkpoint Path should look like
        # `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt|.safetensors`
        assert checkpoint_pt.suffix in {".pt", ".safetensors"}
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json = run_dir / "config.json"
        dataset_statistics_json = _resolve_dataset_statistics(run_dir)
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), (
            f"Missing `dataset_statistics.json` for `{run_dir = }` "
            f"(set STELLAVLA_DATASET_STATISTICS to override)"
        )

        # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`model_id_or_path`)
        # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
        with open(config_json, "r") as f:
            global_cfg = json.load(f)
        _rewrite_base_vlm(global_cfg)

        # Load Dataset Statistics for Action Denormalization
        with open(dataset_statistics_json, "r") as f:
            norm_stats = json.load(f)
    else:
        overwatch.error(f"❌ Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
        raise FileNotFoundError(f"Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
    return global_cfg, norm_stats


def read_mode_config(pretrained_checkpoint):
    """
    Same as read_model_config (legacy duplicate kept for backward compatibility).

    Args:
        pretrained_checkpoint: Path to a .pt checkpoint file.

    Returns:
        tuple:
            vla_cfg (dict)
            norm_stats (dict)
    """
    if os.path.isfile(pretrained_checkpoint):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(pretrained_checkpoint))}`")

        # [Validate] Checkpoint Path should look like
        # `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt|.safetensors`
        assert checkpoint_pt.suffix in {".pt", ".safetensors"}
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_yaml = run_dir / "config.yaml"
        dataset_statistics_json = _resolve_dataset_statistics(run_dir)
        assert config_yaml.exists(), f"Missing `config.yaml` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), (
            f"Missing `dataset_statistics.json` for `{run_dir = }` "
            f"(set STELLAVLA_DATASET_STATISTICS to override)"
        )

        # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`model_id_or_path`)
        # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
        try:
            ocfg = OmegaConf.load(str(config_yaml))
            global_cfg = OmegaConf.to_container(ocfg, resolve=True)
        except Exception as e:
            overwatch.error(f"❌ Failed to load YAML config `{config_yaml}`: {e}")
            raise
        _rewrite_base_vlm(global_cfg)

        # Load Dataset Statistics for Action Denormalization
        with open(dataset_statistics_json, "r") as f:
            norm_stats = json.load(f)
    else:
        overwatch.error(f"❌ Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
        raise FileNotFoundError(f"Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
    return global_cfg, norm_stats

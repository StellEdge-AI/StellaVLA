"""Framework factory.

Every framework module registers itself with `FRAMEWORK_REGISTRY`, and
`build_framework(cfg)` instantiates the one named by `cfg.framework.name`.
"""

import pkgutil
import importlib
from stellavla.model.tools import FRAMEWORK_REGISTRY

from stellavla.utils import initialize_overwatch

logger = initialize_overwatch(__name__)

try:
    pkg_path = __path__
except NameError:
    pkg_path = None

# Auto-import all framework submodules to trigger registration.
# Per-module try/except so one bad import (e.g. optional dep missing like
# snntorch on ST a800) doesn't skip *all subsequent* registrations alphabetically.
if pkg_path is not None:
    for _, module_name, _ in pkgutil.iter_modules(pkg_path):
        try:
            importlib.import_module(f"{__name__}.{module_name}")
        except Exception as e:
            logger.warning(
                f"Failed to auto-import framework submodule '{module_name}': {e}. "
                f"Other frameworks still registered."
            )
        
def build_framework(cfg):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: registry id, e.g. "StellaVLA"
    Returns:
        nn.Module: Instantiated framework model.
    """

    if not hasattr(cfg.framework, "name"):
        cfg.framework.name = cfg.framework.framework_py  # Backward compatibility for legacy config yaml

    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        raise NotImplementedError(
            f"Framework '{cfg.framework.name}' is not implemented. "
            f"Available: {sorted(FRAMEWORK_REGISTRY._registry)}"
        )

    MODLE_CLASS = FRAMEWORK_REGISTRY[framework_id]
    return MODLE_CLASS(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]

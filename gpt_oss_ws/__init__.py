"""GPT-OSS latent workspace package."""

from .config import WorkspaceConfig, load_config

__all__ = [
  "WorkspaceConfig",
  "load_config",
  "GPTOSSHookedModel",
]


def __getattr__(name: str):
  if name == "GPTOSSHookedModel":
    from .model_wrapper import GPTOSSHookedModel as _Model

    return _Model
  raise AttributeError(name)

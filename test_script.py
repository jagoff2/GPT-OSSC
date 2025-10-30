from pathlib import Path
from gpt_oss_ws.config import WorkspaceConfig
from gpt_oss_ws.model_wrapper import GPTOSSHookedModel

cfg_path = Path('configs/server.yaml')
cfg = WorkspaceConfig.from_yaml(cfg_path)
print('model dtype', next(GPTOSSHookedModel(cfg).model.parameters()).dtype)

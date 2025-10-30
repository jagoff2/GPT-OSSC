import torch
from gpt_oss_ws.config import WorkspaceConfig
from gpt_oss_ws.model_wrapper import GPTOSSHookedModel
from gpt_oss_ws.types import GenerationRequestContext, HookToggles
from gpt_oss_ws.generation import generate_with_workspace

cfg = WorkspaceConfig(
    model_name='openai/gpt-oss-20b',
    quantization='fp32',
    device_map='auto',
    hooked_layers=[1,5,9,13,17,21],
    nvirt=4,
    residual_rank=8,
    slot_count=6,
    slot_dim=256,
    slot_iterations=1,
    enable_kv_append=True,
    enable_residual_delta=True,
    enable_read_probes=True,
    enable_broadcast=True,
    workspace_device='cpu',
    retention=None,
    log_level='INFO',
    api_host='0.0.0.0',
    api_port=8000,
    controller_entropy_floor=2.5,
    controller_norm_cap=4.5,
    sqlite_path='workspace_memory.sqlite',
    faiss_index_path='workspace_memory.faiss',
    memory_embedding_dim=384,
    max_context_tokens=8192,
    bf16_fallback=True,
)
model = GPTOSSHookedModel(cfg)
request = GenerationRequestContext(request_id='1', toggles=HookToggles(True,True,True,True))
input_ids = model.tokenizer_encode('Hello')
print('input_ids', input_ids.shape)
out = generate_with_workspace(model, request, input_ids, max_new_tokens=1)
print('output shape', out.shape)


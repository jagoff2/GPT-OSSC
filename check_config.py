from gpt_oss_ws.config import load_config
cfg = load_config('G:\\ossc\\configs\\server.yaml')
print('retention', cfg.retention)
print('virt_kv_max_tokens_per_layer', cfg.retention.virt_kv_max_tokens_per_layer)


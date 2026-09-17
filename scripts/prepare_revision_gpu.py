"""Download the public model into a task-local directory and record provenance."""
import importlib.metadata
import json
import platform
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root = Path(__file__).resolve().parents[1]
out = root / 'results/revision'
out.mkdir(parents=True, exist_ok=True)
model_id = 'Qwen/Qwen3-4B'
revision = HfApi(token=False).model_info(model_id).sha
metadata = {
    'model_id': model_id, 'model_revision': revision,
    'platform': platform.platform(), 'python': platform.python_version(),
    'packages': {name: importlib.metadata.version(name) for name in
                 ['torch', 'transformers', 'peft', 'trl', 'datasets', 'accelerate', 'bitsandbytes']},
    'purpose': 'New-platform replication; original results remain unchanged',
}
(out/'environment.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
print(json.dumps(metadata), flush=True)
snapshot_download(model_id, revision=revision, token=False, local_dir=root/'model',
                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'], max_workers=3)
print('MODEL_READY', flush=True)

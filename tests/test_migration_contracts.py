"""检查仓库路径及迁移后的独立进程导入，禁止访问正式结果。"""
import importlib
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_m1_contracts import utils


ROOT = Path(__file__).resolve().parents[1]
MIGRATED = ROOT / "src/scripts"


def test_repository_relative_paths_remain_rooted_at_repository(tmp_path):
    assert utils.REPO_ROOT == ROOT
    assert utils.abspath("configs/m1_llama2_7b.yaml") == ROOT / "configs/m1_llama2_7b.yaml"
    assert utils.abspath(tmp_path) == tmp_path
    path = tmp_path / "config.yaml"
    path.write_text("adapter:\n  r: 128\n  niter: null\n", encoding="utf-8")
    assert utils.load_config(path) == {"adapter": {"r": 128, "niter": None}}


@pytest.mark.parametrize("name,dtype", [("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)])
def test_dtype_mapping_is_lookup(name, dtype):
    assert utils.torch_dtype(name) == dtype
    with pytest.raises(KeyError):
        utils.torch_dtype("unsupported")


@pytest.mark.skipif(not MIGRATED.exists(), reason="迁移前先建立行为基线")
def test_normal_imports_do_not_load_quantization_backend():
    code = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from scripts import model_ops, data_ops
from lora import LoRALinear
from pissa import PiSSALinear
assert 'bitsandbytes' not in sys.modules
assert 'pissa.layer_nf4' not in sys.modules
assert model_ops.LoRALinear is LoRALinear
assert model_ops.PiSSALinear is PiSSALinear
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not MIGRATED.exists(), reason="迁移前先建立行为基线")
@pytest.mark.parametrize("module", ["scripts.m0.verify_init", "scripts.m0.mnist_toy", "scripts.m0.plot_singular", "scripts.m1.train", "scripts.m1.prepare_data", "scripts.m1.eval_gsm8k", "scripts.m1.plot_curves", "scripts.plot.regenerate_figures"])
def test_migrated_modules_are_importable_without_running_main(module):
    imported = importlib.import_module(module)
    assert imported.__file__ and str(MIGRATED) in imported.__file__

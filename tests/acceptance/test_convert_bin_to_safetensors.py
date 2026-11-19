"""convert_bin_to_safetensors：小型假分片的无损转换、index 重写与失败路径。"""
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "conv", ROOT / "src" / "scripts" / "download" / "convert_bin_to_safetensors.py")
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)


def _make(tmp_path):
    g = torch.Generator().manual_seed(0)
    s1 = {"a.weight": torch.randn(4, 3, generator=g).bfloat16(), "b.weight": torch.randn(5, generator=g).bfloat16()}
    s2 = {"c.weight": torch.randn(2, 2, generator=g).bfloat16()}
    torch.save(s1, tmp_path / "pytorch_model-00001-of-00002.bin")
    torch.save(s2, tmp_path / "pytorch_model-00002-of-00002.bin")
    wm = {**{k: "pytorch_model-00001-of-00002.bin" for k in s1}, **{k: "pytorch_model-00002-of-00002.bin" for k in s2}}
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        json.dumps({"metadata": {"total_size": 123}, "weight_map": wm}))
    return s1, s2


def test_st_name():
    assert conv.st_name("pytorch_model-00001-of-00002.bin") == "model-00001-of-00002.safetensors"
    with pytest.raises(ValueError):
        conv.st_name("foo.bin")


def test_convert_lossless_and_index(tmp_path):
    s1, s2 = _make(tmp_path)
    outs = conv.convert(tmp_path)
    assert [o.name for o in outs] == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    back = load_file(str(outs[0]))
    for k, v in s1.items():
        assert back[k].dtype == torch.bfloat16 and torch.equal(back[k], v)
    idx = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert idx["metadata"] == {"total_size": 123}
    assert idx["weight_map"]["c.weight"] == "model-00002-of-00002.safetensors"
    assert (tmp_path / "pytorch_model-00001-of-00002.bin").exists()  # 原 .bin 保留


def test_refuses_to_overwrite_and_missing_files(tmp_path):
    _make(tmp_path)
    conv.convert(tmp_path)
    with pytest.raises(FileExistsError):
        conv.convert(tmp_path)
    conv.convert(tmp_path, overwrite=True)
    (tmp_path / "pytorch_model-00002-of-00002.bin").unlink()
    with pytest.raises(FileNotFoundError):
        conv.convert(tmp_path, overwrite=True)

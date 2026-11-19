"""把 HF 分片的 ``pytorch_model-*.bin`` 无损转成 ``model-*.safetensors``（同目录，原 .bin 保留）。

背景：transformers 5.x 只查找 safetensors，不再回退加载 ``.bin``；ModelScope 的
``AI-ModelScope/Mistral-7B-v0.1`` 只有 ``.bin``。

做法：逐分片 ``torch.load(weights_only=True)`` → ``safetensors.torch.save_file``，
分片文件名与张量分组沿用 ``pytorch_model.bin.index.json``；写出 ``model.safetensors.index.json``。
转换后重新读取，对每个张量比较 dtype、shape 和逐元素相等，任何不一致都会报错退出
（不会写出新的 index 文件）。

不修改 .bin、config、tokenizer 文件；转换结果不是官方 safetensors 的逐字节副本，
只保证张量内容与 .bin 完全相同。

用法：
    python src/scripts/download/convert_bin_to_safetensors.py --model-dir <含 pytorch_model.bin.index.json 的目录>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

BIN_INDEX = "pytorch_model.bin.index.json"
ST_INDEX = "model.safetensors.index.json"


def st_name(bin_name: str) -> str:
    """``pytorch_model-00001-of-00002.bin`` → ``model-00001-of-00002.safetensors``。"""
    if not (bin_name.startswith("pytorch_model") and bin_name.endswith(".bin")):
        raise ValueError(f"无法识别的分片名：{bin_name}")
    return "model" + bin_name[len("pytorch_model"):-len(".bin")] + ".safetensors"


def convert(model_dir: Path, overwrite: bool = False) -> list[Path]:
    """转换并校验 ``model_dir`` 下的全部分片。

    Args:
        model_dir: 含 ``pytorch_model.bin.index.json`` 及其分片的目录。
        overwrite: 目标 safetensors 已存在时是否覆盖；默认报错退出。

    Returns:
        写出的 safetensors 文件路径列表（不含 index）。

    Raises:
        FileNotFoundError: 缺少 index 或分片。
        FileExistsError: 目标已存在且未设置 overwrite。
        RuntimeError: 转换后的张量与 .bin 的 dtype/shape/数值不一致，或张量集合不匹配。
    """
    model_dir = Path(model_dir)
    index_path = model_dir / BIN_INDEX
    if not index_path.is_file():
        raise FileNotFoundError(f"找不到 {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    bin_files = sorted(set(weight_map.values()))
    for b in bin_files:
        if not (model_dir / b).is_file():
            raise FileNotFoundError(f"缺少分片 {model_dir / b}")
        if (model_dir / st_name(b)).exists() and not overwrite:
            raise FileExistsError(f"目标已存在：{model_dir / st_name(b)}（如需覆盖加 --overwrite）")

    outputs: list[Path] = []
    new_map: dict[str, str] = {}
    for b in bin_files:
        out = model_dir / st_name(b)
        print(f"[convert] {b} → {out.name}", flush=True)
        state = torch.load(model_dir / b, map_location="cpu", weights_only=True)
        expected = {k for k, v in weight_map.items() if v == b}
        if set(state) != expected:
            raise RuntimeError(f"{b} 的张量名与 index 不一致：多 {set(state) - expected}，少 {expected - set(state)}")
        # safetensors 要求连续且互不共享内存
        tensors = {k: v.contiguous().clone() for k, v in state.items()}
        save_file(tensors, str(out), metadata={"format": "pt"})
        del tensors

        back = load_file(str(out), device="cpu")
        if set(back) != set(state):
            raise RuntimeError(f"{out.name} 读回的张量名不一致")
        for k, v in state.items():
            w = back[k]
            if w.dtype != v.dtype or w.shape != v.shape or not torch.equal(w, v):
                raise RuntimeError(f"{out.name} 的张量 {k} 与 .bin 不一致（dtype/shape/数值）")
            new_map[k] = out.name
        print(f"[verify] {out.name}: {len(state)} 个张量逐元素一致", flush=True)
        del state, back
        outputs.append(out)

    new_index = {"metadata": index.get("metadata", {}), "weight_map": new_map}
    (model_dir / ST_INDEX).write_text(json.dumps(new_index, indent=2), encoding="utf-8")
    print(f"[done] 写出 {ST_INDEX}", flush=True)
    return outputs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    try:
        convert(Path(args.model_dir), overwrite=args.overwrite)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

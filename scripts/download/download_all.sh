#!/usr/bin/env bash
# 下载并准备全部数据与模型（M1 / M2 / M3 共用）。
#
# 内容：
#   1. MetaMathQA 前 100K 条 + GSM8K test（经 scripts/m1/prepare_data.py 落成本地 jsonl）
#   2. LLaMA-2-7B           -> M1 / M3
#   3. ChatGLM3-6B-Base     -> M2
#   4. Baichuan2-7B-Base    -> M2
#   5. Qwen2.5-7B           -> M2
#   6. Meta-Llama-3-8B      -> M2（跳过 original/ 下 16GB 的 .pth 副本）
#
# 用法（建议在 screen/tmux 里，约需 70GB 磁盘；网速一般时全部下完需数小时）：
#   bash scripts/download/download_all.sh 2>&1 | tee -a download_all.log
#
# 前置：已按 requirements.txt 装好依赖（需要 modelscope 及其命令行 `modelscope`）。
#
# 说明：
#   - 不加 set -e：某一步失败不影响后续步骤；每步结束会打印 exit 码，
#     事后可 grep -nE "Traceback|Killed|Error|exit=" download_all.log 定位。
#   - 所有目录与 configs/*.yaml 中的路径一一对应，不要随意改名：
#       configs/m1_llama2_7b.yaml  : model.cache_dir = data/m1/models（snapshot_download 自动建子目录）
#       configs/m2_*.yaml          : model.local_path = data/m2/models/...（下面 --local-dir 的值）
#       configs/m3_llama2_7b.yaml  : model.local_path = data/m1/models/models/modelscope--Llama-2-7b-ms/snapshots/master
#     （即 M1 的 snapshot_download 实际落点，M1 与 M3 共用同一份 LLaMA-2 权重。）
#   - 下载脚本是幂等的：已下载完整的文件会被跳过，中断后重跑即可续传。

# 缓存放数据盘，避免写满系统盘
export MODELSCOPE_CACHE=/root/autodl-tmp/.cache/modelscope
export HF_HOME=/root/autodl-tmp/.cache/huggingface

# 定位仓库根目录（本脚本位于 <repo>/scripts/download/）
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1

stamp() { echo; echo "===== [$(date '+%F %T')] $* ====="; }

stamp "1/6 数据：MetaMathQA 前 100K + GSM8K test"
python scripts/m1/prepare_data.py --config configs/m1_llama2_7b.yaml
echo "prepare_data exit=$?"

stamp "2/6 LLaMA-2-7B（参数与 common.resolve_model_path 一致，训练时直接命中缓存）"
python -c "
from modelscope import snapshot_download
p = snapshot_download('modelscope/Llama-2-7b-ms', cache_dir='$REPO/data/m1/models', ignore_patterns=['*.bin', '*.pth', '*.pdf'])
print('LLaMA-2 path:', p)
"
echo "llama2 exit=$?"

stamp "3/6 ChatGLM3-6B-Base"
modelscope download ZhipuAI/chatglm3-6b-base --local-dir "$REPO/data/m2/models/ChatGLM3-6B-Base/ZhipuAI/chatglm3-6b-base"
echo "chatglm3 exit=$?"

stamp "4/6 Baichuan2-7B-Base"
modelscope download baichuan-inc/Baichuan2-7B-Base --local-dir "$REPO/data/m2/models/Baichuan2-7B-Base/baichuan-inc/Baichuan2-7B-Base"
echo "baichuan2 exit=$?"

stamp "5/6 Qwen2.5-7B"
modelscope download Qwen/Qwen2.5-7B --local-dir "$REPO/data/m2/models/Qwen2.5-7B/Qwen/Qwen2___5-7B"
echo "qwen25 exit=$?"

stamp "6/6 Meta-Llama-3-8B（跳过 original/*）"
modelscope download LLM-Research/Meta-Llama-3-8B --local-dir "$REPO/data/m2/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B" --exclude "original/*"
echo "llama3 exit=$?"

stamp "汇总"
wc -l data/m1/*.jsonl
# 第 2 项是 snapshot_download 的实际落点；其余是 --local-dir
for d in data/m1/models/models/modelscope--Llama-2-7b-ms/snapshots/master \
         data/m2/models/ChatGLM3-6B-Base/ZhipuAI/chatglm3-6b-base \
         data/m2/models/Baichuan2-7B-Base/baichuan-inc/Baichuan2-7B-Base \
         data/m2/models/Qwen2.5-7B/Qwen/Qwen2___5-7B \
         data/m2/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B; do
  printf "%-95s %6s  weight_files=%s\n" "$d" "$(du -shL "$d" 2>/dev/null | cut -f1)" \
    "$(ls "$d" 2>/dev/null | grep -cE '\.(safetensors|bin)$')"
done
df -h "$REPO"
stamp "全部结束"

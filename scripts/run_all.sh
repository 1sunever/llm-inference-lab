#!/usr/bin/env bash
# 一键复现全部实验。
#
# 环境假设：Linux（本仓库在 WSL2 Ubuntu 24.04 上验证），PATH 里的 python
# 已装好 torch>=2.4 与 triton>=3.0，且能访问 GPU。
# 若用别的解释器，改下面这一行即可。
set -u
PY="${PY:-python3}"
cd "$(dirname "$0")/.." || exit 1

echo "=============================================================="
echo " llm-inference-lab · 复现全部实验"
echo " python: $($PY -c 'import sys;print(sys.executable)')"
echo "=============================================================="
$PY -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__, '| cuda', torch.cuda.is_available())" || {
  echo "依赖缺失：pip install -r requirements.txt"; exit 1; }

FAIL=0
run() {
  local name="$1"; shift
  echo
  echo "--------------------------------------------------------------"
  echo ">> $name"
  echo "--------------------------------------------------------------"
  if "$@"; then
    echo "   [ok] $name"
  else
    echo "   [FAIL] $name"
    FAIL=$((FAIL + 1))
  fi
}

# 0) 正确性优先：算错了的 benchmark 没有意义
run "flash_attn 正确性 (12 用例)"  $PY -X utf8 tests/test_flash_attn.py

# 1) 四个模块的实验（各自会写 results/*.json 并画 figures/*.png）
run "flash_attn 基准"   $PY -X utf8 src/llm_lab/flash_attn/bench.py
run "paged_attn 基准"   $PY -X utf8 src/llm_lab/paged_attn/bench.py
run "moe 基准"          $PY -X utf8 src/llm_lab/moe/bench.py
run "spec_decode 基准"  $PY -X utf8 src/llm_lab/spec_decode/bench.py

echo
echo "=============================================================="
if [ "$FAIL" -eq 0 ]; then
  echo " 全部完成。产物：results/*.json  figures/*.png"
else
  echo " 有 $FAIL 项失败，见上方日志。"
fi
echo "=============================================================="
exit "$FAIL"
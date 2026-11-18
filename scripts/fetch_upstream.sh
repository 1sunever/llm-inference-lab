#!/usr/bin/env bash
# 拉取上游参考实现，用于对照阅读。
#
# 为什么要这个脚本而不是直接 vendoring：
#   把 vLLM/flash-attention 的源码拷进本仓库，会带来许可证头维护成本，
#   也会让"哪部分是我写的"变得含糊。这里改成按需拉取到 _upstream/
#   （已 gitignore），读的时候对着 third_party/README.md 的表格找文件。
#
# 用法：
#   bash scripts/fetch_upstream.sh            # 浅克隆全部
#   bash scripts/fetch_upstream.sh vllm       # 只拉某一个
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/_upstream"
mkdir -p "$DEST"

# 注意：--depth 1 只取主分支快照，够读源码；要按 tag 精确对应版本时
# 用 --branch <tag>。vLLM 的 v1 架构在 0.32 前后差别不小，
# 对照前先确认你读的是哪个版本。
repos=(
  "flash-attention|https://github.com/Dao-AILab/flash-attention.git"
  "vllm|https://github.com/vllm-project/vllm.git"
  "triton|https://github.com/triton-lang/triton.git"
)

want="${1:-all}"

for entry in "${repos[@]}"; do
  name="${entry%%|*}"
  url="${entry##*|}"
  if [ "$want" != "all" ] && [ "$want" != "$name" ]; then
    continue
  fi
  if [ -d "$DEST/$name/.git" ]; then
    echo "[skip] $name 已存在"
    continue
  fi
  echo "[clone] $name <- $url"
  if ! git clone --depth 1 --single-branch -q "$url" "$DEST/$name"; then
    echo "[warn] $name 克隆失败（网络原因？国内的 release 下载常被拦，"
    echo "       但 git 协议一般可用；若失败可尝试配置代理或换镜像）"
    continue
  fi
  echo "        $(git -C "$DEST/$name" log -1 --format='%h %cs')"
done

echo
echo "完成。上游代码在 $DEST"
echo "对照表见 third_party/README.md"
echo
echo "如果只想确认本机装的 vLLM 版本对应哪份源码："
echo "  python -c \"import vllm, os; print(vllm.__version__, os.path.dirname(vllm.__file__))\""
echo "  # 站点包里的 vllm 目录本身就是完整源码，不必再克隆一份"
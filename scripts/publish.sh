#!/usr/bin/env bash
# 一键发布到 GitHub。
#
# 背景：这个仓库做完了但发布被网络卡住过——本机对 github.com 的 HTTPS 不稳定，
# `gh auth login --web` 需要的 OAuth POST 会超时；而 SSH 是通的。
# 所以这里给两条互斥的路径，脚本自动判断走哪条。
#
#   A) 已 `gh auth login`（或用 PAT：`gh auth login --with-token`）
#      -> 脚本建仓库并推送
#   B) 还没 gh 凭据，但仓库已在网页上建好（空的，不勾 README/license）
#      -> 脚本直接 push（走 SSH，需 ~/.ssh 已注册到 GitHub）
#
# 用法：
#   bash scripts/publish.sh
#   bash scripts/publish.sh --dry-run     # 只检查，不推送
set -u

REPO_SLUG="1sunever/llm-inference-lab"
BRANCH="main"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

cd "$(dirname "$0")/.." || exit 1

echo "== 当前状态 =="
echo "分支: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
echo "提交: $(git log --oneline -1 2>/dev/null || echo '?')"
echo "未提交改动: $(git status --porcelain | wc -l) 个文件"
git remote -v | sed 's/^/  /'

have_gh=0
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  have_gh=1
fi

echo
echo "== 选择路径 =="
if [ "$have_gh" -eq 1 ]; then
  echo "检测到 gh 已登录 -> 用 gh 建仓库并推送"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] 将执行: gh repo create $REPO_SLUG --public --source=. --remote=origin --push"
    exit 0
  fi
  gh repo create "$REPO_SLUG" --public --source=. --remote=origin --push \
    && echo "完成：https://github.com/$REPO_SLUG" \
    || { echo "gh repo create 失败（仓库可能已存在），退回直接 push"; git push -u origin "$BRANCH"; }
  exit $?
fi

echo "gh 未登录 -> 走「仓库已建好 + SSH 直接 push」"
echo
echo "前置（只需做一次）："
echo "  1. 浏览器打开 https://github.com/new?name=llm-inference-lab"
echo "  2. 选 Public，**不要**勾 Add README / .gitignore / license"
echo "  3. 点 Create repository"
echo

case "$(git remote get-url origin 2>/dev/null || true)" in
  "") echo "未配置 origin，自动添加 SSH 远端"
      [ "$DRY_RUN" -eq 1 ] && { echo "[dry-run] git remote add origin git@github.com:$REPO_SLUG.git"; exit 0; }
      git remote add origin "git@github.com:$REPO_SLUG.git" ;;
  *)  ;;
esac

# 检查 SSH 是否能通过（GitHub 对 SSH 的认证探测，成功时会打印 Hi <user>）
echo "== 检查 SSH 认证 =="
if ! ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
  echo "SSH 认证失败：确认 ~/.ssh/id_ed25519 已加到 GitHub（Settings -> SSH and GPG keys）"
  echo "或在有网络时改用：gh auth login --with-token < token.txt"
  exit 1
fi
echo "SSH OK"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] 将执行: git push -u origin $BRANCH"
  exit 0
fi

git push -u origin "$BRANCH" && echo "完成：https://github.com/$REPO_SLUG"
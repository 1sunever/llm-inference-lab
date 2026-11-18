<#
.SYNOPSIS
    把 llm-inference-lab 发布到 GitHub。

.DESCRIPTION
    仓库内容已就绪，但发布需要一次外部授权。本机对 github.com 的 HTTPS 不稳定
    （gh auth login --web 所需的 OAuth POST 会超时），而 SSH 是通的。
    因此这里给两条互斥路径，脚本自动判断：

      A) gh 已登录（gh auth login / --with-token）
         -> 建仓库并推送
      B) 没有 gh 凭据，但空仓库已在网页建好
         -> 直接 git push（走 SSH，用 %USERPROFILE%\.ssh\id_ed25519）

    注意：WSL 里没有 SSH key，Windows 这把才有。所以请在 Windows 侧运行本脚本。

.EXAMPLE
    # 先看会做什么（不推送）
    powershell -ExecutionPolicy Bypass -File scripts\publish.ps1 -DryRun

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\publish.ps1
#>
param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Continue'
$RepoSlug = '1sunever/llm-inference-lab'
$Branch   = 'main'

Push-Location (Join-Path $PSScriptRoot '..')
try {
    Write-Host '== 当前状态 ==' -ForegroundColor Cyan
    Write-Host ('分支: '   + (& git rev-parse --abbrev-ref HEAD))
    Write-Host ('提交: '   + (& git log --oneline -1))
    $dirty = (& git status --porcelain | Measure-Object -Line).Lines
    Write-Host ('未提交改动: ' + $dirty + ' 个文件')
    & git remote -v | ForEach-Object { '  ' + $_ }

    # gh 是否可用且已登录
    $haveGh = $false
    $ghCmd = Get-Command gh -ErrorAction SilentlyContinue
    if ($ghCmd) {
        & gh auth status *> $null
        $haveGh = ($LASTEXITCODE -eq 0)
    }

    Write-Host ''
    Write-Host '== 选择路径 ==' -ForegroundColor Cyan

    if ($haveGh) {
        Write-Host '检测到 gh 已登录 -> 用 gh 建仓库并推送'
        # 注意不能叫 $args：那是 PowerShell 的自动变量，赋值会报只读错误
        $ghArgs = @('repo', 'create', $RepoSlug, '--public', '--source=.', '--remote=origin', '--push')
        if ($DryRun) { Write-Host ('[dry-run] gh ' + ($ghArgs -join ' ')); return }
        & gh @ghArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host 'gh repo create 失败（仓库可能已存在），退回直接 push'
            & git push -u origin $Branch
        }
        return
    }

    Write-Host 'gh 未登录 -> 走「空仓库已建好 + SSH 直接 push」'
    Write-Host ''
    Write-Host '前置（只需做一次）：'
    Write-Host '  1. 浏览器打开 https://github.com/new?name=llm-inference-lab'
    Write-Host '  2. 选 Public，不要勾 Add README / .gitignore / license'
    Write-Host '  3. 点 Create repository'
    Write-Host ''

    $origin = ''
    & git remote get-url origin *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host ('未配置 origin，添加 SSH 远端 git@github.com:' + $RepoSlug + '.git')
        if ($DryRun) { Write-Host ('[dry-run] git remote add origin git@github.com:' + $RepoSlug + '.git') }
        else { & git remote add origin ('git@github.com:' + $RepoSlug + '.git') }
    } else {
        $origin = (& git remote get-url origin).Trim()
        Write-Host ('已有 origin: ' + $origin)
    }

    Write-Host '== 检查 SSH 认证 =='
    $sshOut = & ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -T git@github.com 2>&1 | Out-String
    if ($sshOut -notmatch 'successfully authenticated') {
        Write-Host 'SSH 认证失败。检查：' -ForegroundColor Yellow
        Write-Host '  - %USERPROFILE%\.ssh\id_ed25519 是否存在'
        Write-Host '  - 该公钥是否已加到 GitHub（Settings -> SSH and GPG keys）'
        Write-Host ('  输出: ' + $sshOut.Trim())
        exit 1
    }
    Write-Host 'SSH OK'

    if ($DryRun) { Write-Host ('[dry-run] git push -u origin ' + $Branch); return }

    & git push -u origin $Branch
    if ($LASTEXITCODE -eq 0) {
        Write-Host ('完成: https://github.com/' + $RepoSlug) -ForegroundColor Green
    }
} finally {
    Pop-Location
}
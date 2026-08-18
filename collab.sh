#!/usr/bin/env bash
# collab.sh — research-stack herdr 协作助手
# ---------------------------------------------------------------------------
# 一个文件替代旧三件套: .pi-pane(自报 pane) + tmux send-keys(派活) + tmux-status.sh(观测)
#
# 自发现规则: peer = herdr agent list 里 workspace 相同 ∧ cwd 相同(项目根) ∧ kind 对端
#   - student(kind=pi) 的 peer 是 claude; teacher(kind=claude) 的 peer 是 pi
#   - 两 agent 都 cd 到项目根 → "同 workspace + 同 cwd" 天然锁定本项目 partner,零歧义
#
# 依赖: herdr(在 PATH)、python3。须在项目根运行(与两 agent 的 cwd 一致)。
#
# 用法:
#   ./collab.sh self                      # 打印本 agent 的 pane id
#   ./collab.sh peer [pi|claude]          # 打印 peer 的 pane id(缺省=我的对端 kind)
#   ./collab.sh send <kind> "<msg>" [--wait] [--timeout MS]
#                                         # 发现 peer(kind=pi|claude) + herdr agent prompt 投递
#   ./collab.sh status                    # 本项目两 agent 的生命周期状态(替代 tmux-status.sh)
set -euo pipefail

require_herdr() {
  [ "${HERDR_ENV:-}" = 1 ] || { echo "ERR: 不在 herdr (HERDR_ENV!=1)。tmux 模式请用旧 .pi-pane + tmux send-keys。" >&2; exit 1; }
  command -v herdr >/dev/null 2>&1 || { echo "ERR: herdr 不在 PATH" >&2; exit 1; }
  command -v python3 >/dev/null 2>&1 || { echo "ERR: python3 不在 PATH(collab.sh 用它解析 herdr JSON)" >&2; exit 1; }
}

# 用 python3 解析 herdr agent list。env: HERDR_WORKSPACE_ID / COLLAB_CWD(=调用方 $PWD)
# $1=mode: "match" 按 kind 过滤打印 pane_id; "self" 打印本 pane 的 kind; "status" 打印表格
_py() {
  HERDR_WORKSPACE_ID="${HERDR_WORKSPACE_ID:-}" \
  COLLAB_CWD="$PWD" \
  HERDR_PANE_ID="${HERDR_PANE_ID:-}" \
  python3 - "$@" <<'PY'
import json, os, sys, subprocess
mode = sys.argv[1]
ws   = os.environ.get("HERDR_WORKSPACE_ID","")
cwd  = os.environ.get("COLLAB_CWD","")
me   = os.environ.get("HERDR_PANE_ID","")
try:
    out = subprocess.run(["herdr","agent","list"], capture_output=True, text=True)
    agents = json.loads(out.stdout).get("result", {}).get("agents", [])
except Exception as e:
    sys.stderr.write(f"ERR: 解析 herdr agent list 失败: {e}\n"); sys.exit(1)

def mine():
    for a in agents:
        if a.get("pane_id") == me: return a
    return None

if mode == "self":
    a = mine(); print(a.get("agent","") if a else ""); sys.exit(0)

if mode == "status":
    rows = [a for a in agents if a.get("workspace_id")==ws and a.get("cwd")==cwd]
    print(f"{'PANE':<10} {'KIND':<8} {'STATUS':<10} {'TAB':<10} CWD")
    for a in rows:
        print(f"{a.get('pane_id',''):<10} {a.get('agent',''):<8} {a.get('agent_status',''):<10} {a.get('tab_id',''):<10} {a.get('cwd','')}")
    if not rows: sys.stderr.write(f"(本项目无 agent: workspace={ws} cwd={cwd})\n")
    sys.exit(0)

# mode == "match": argv[2]=want_kind → 打印匹配 pane_id(每行一个),排除自己
want = sys.argv[2]
cands = [a for a in agents
         if a.get("workspace_id")==ws and a.get("cwd")==cwd
         and a.get("agent")==want and a.get("pane_id")!=me]
for a in cands: print(a.get("pane_id",""))
PY
}

opposite_kind() { case "$1" in pi) echo claude;; claude) echo pi;; *) echo "$1";; esac; }

cmd_self() { require_herdr; echo "${HERDR_PANE_ID:-}"; }

cmd_mykind() { require_herdr; _py self; }

# peer [kind]: 缺省 kind = 我的对端 kind
cmd_peer() {
  require_herdr
  local kind="${1:-}"
  if [ -z "$kind" ]; then
    kind="$(opposite_kind "$(cmd_mykind)")"
  fi
  local panes; panes="$(_py match "$kind" || true)"
  if [ -z "$panes" ]; then
    echo "ERR: 未找到 kind=$kind 的 peer(同 workspace=$HERDR_WORKSPACE_ID cwd=$PWD)。" >&2
    echo "    确认 partner agent 已在本项目目录启动,且 herdr 已识别它(状态非 unknown)。" >&2
    exit 1
  fi
  local n; n="$(printf '%s\n' "$panes" | grep -c . || true)"
  if [ "$n" -gt 1 ]; then
    echo "WARN: 多个 $kind peer,取第一个。全部: $(printf '%s\n' "$panes" | tr '\n' ' ')" >&2
  fi
  printf '%s\n' "$panes" | head -1
}

# send <kind> "<msg>" [--wait] [--timeout MS]
cmd_send() {
  require_herdr
  local kind="$1"; shift
  local msg="$1"; shift
  local wait=0 timeout=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --wait) wait=1; shift;;
      --timeout) timeout="${2:-}"; shift 2;;
      *) echo "ERR: 未知参数 $1" >&2; exit 1;;
    esac
  done
  [ -n "$msg" ] || { echo "ERR: 消息为空" >&2; exit 1; }
  local peer; peer="$(cmd_peer "$kind")"
  local args=(herdr agent prompt "$peer" "$msg")
  [ "$wait" = 1 ] && args+=(--wait)
  [ -n "$timeout" ] && args+=(--timeout "$timeout")
  echo "→ 投递给 $kind ($peer): $msg" >&2
  "${args[@]}"
}

cmd_status() { require_herdr; _py status; }

case "${1:-}" in
  self)   cmd_self;;
  mykind) cmd_mykind;;
  peer)   shift; cmd_peer "$@";;
  send)   shift; cmd_send "$@";;
  status) cmd_status;;
  ""|-h|--help|help)
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | grep -v '^set -euo' ;;
  *) echo "ERR: 未知命令 $1" >&2; exit 1;;
esac

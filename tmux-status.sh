#!/usr/bin/env bash
# meanflowse 协作状态面板 —— tmux 第二 pane 里跑:  watch -n5 ./tmux-status.sh
# 人类一眼看到:当前任务 / 最近完成 / idea 状态 / ledger 修改时间 / pi 是否在跑
# 用法: tmux split-window -h 'watch -n5 ./tmux-status.sh'

set +e
cd "$(dirname "$0")"

B='\033[1m'; R='\033[31m'; G='\033[32m'; Y='\033[33m'; C='\033[36m'; N='\033[0m'

clear
echo -e "${C}===== meanflowse 协作状态  $(date '+%Y-%m-%d %H:%M:%S') =====${N}"
echo

# 1. 当前任务
echo -e "${B}▶ 当前任务 (NEXT_TASK.md IN_PROGRESS/QUEUED)${N}"
sed -n '/## 🔴 IN_PROGRESS/,/^## ⚪ DONE/p' NEXT_TASK.md 2>/dev/null \
  | grep -E '^\-|QUEUED|IN_PROGRESS' | head -8
echo

# 2. 最近完成
echo -e "${G}✓ 最近完成 (DONE_LOG.md 尾部)${N}"
tail -12 DONE_LOG.md 2>/dev/null | grep -E '^## TASK|^- 结果|^- 完成' | head -6
echo

# 3. Idea 状态计数
echo -e "${Y}✎ Ideas (IDEAS.md)${N}"
# 只数真实条目(排除含 < 或 / 的模板行)
real=$(grep -E '^- 状态:' IDEAS.md 2>/dev/null | grep -v '<' | grep -v '/' | wc -l)
echo "  真实 idea 总数: ${real:-0}"
for s in DRAFT REVIEWED APPROVED RUNNING DONE; do
  n=$(grep -E '^- 状态:' IDEAS.md 2>/dev/null | grep -v '<' | grep -v '/' | grep -c "$s")
  [ "${n:-0}" -gt 0 ] && echo "    $s: $n"
done
echo "  待审(未过 novelty gate): $(grep -E '^- novelty-reviewer 判定:' IDEAS.md 2>/dev/null | grep -v '<' | grep -c '未审')"
echo

# 4. 数据真源新鲜度
echo -e "${B}▣ RESULTS_LEDGER.md (唯一数据真源)${N}"
ls -la --time-style='+%m-%d %H:%M' RESULTS_LEDGER.md 2>/dev/null | awk '{print "  最后修改:", $6, $7}'
echo

# 5. pi 进程是否在跑
echo -e "${B}⟳ pi agent 进程${N}"
if pgrep -f 'pi.*sflow\|pi-sflow\|pi-coding' >/dev/null 2>&1; then
  echo -e "  ${G}● pi 在运行${N}"
else
  echo -e "  ${R}○ pi 未检测到(可能空闲或需启动)${N}"
fi
echo
echo -e "${C}按 Ctrl+C 退出; tmux attach pi-sflow 可看 pi 实时干活${N}"

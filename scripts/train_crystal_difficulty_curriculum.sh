#!/usr/bin/env bash
# Difficulty-ramp curriculum for procedurally-generated Crystal Caves.
#
# The agent learns the full clear far faster by mastering a trivially-winnable
# floor first, then ramping the objective/threat load. Each stage warm-starts
# from the previous stage's BEST policy (the trainer auto-loads
# models/crystal_caves/crystal_caves_best.pth on launch), so winning behaviour
# carries upward instead of being relearned cold.
#
#   tutorial : 1 open crystal, no lock/threats  -> get + solidify the first wins
#   easy     : 2-3 crystals, 1 switch-gated lock
#   normal   : full game (10-14 crystals, hazards, colour-keyed locks)
#
#   FRESH=0 scripts/train_crystal_difficulty_curriculum.sh   # continue from best
#   TUT_EPS=600 EASY_EPS=500 NORMAL_EPS=500 ...               # episodes per stage
#
# PYTHONUNBUFFERED keeps the per-episode log streaming (so the per-stage win
# rate is visible and not lost if the run is time-boxed).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

FAM="${FAM:-platform_network}"

if [[ "${FRESH:-1}" == "1" ]]; then
  echo "🧹 fresh start: clearing models/crystal_caves/*.pth"
  rm -f models/crystal_caves/*.pth
else
  echo "↪️  continuing from existing best policy (FRESH=0)"
fi

run_stage () {
  local diff="$1" eps="$2"
  echo ""
  echo "================ DIFFICULTY STAGE: ${diff} (${eps} episodes) ================"
  python main.py --headless --game crystal_caves --random-caves \
    --cave-difficulty "$diff" --cave-families "$FAM" \
    --turbo --vec-envs 8 --episodes "$eps"
}

run_stage tutorial "${TUT_EPS:-600}"
run_stage easy "${EASY_EPS:-500}"
run_stage normal "${NORMAL_EPS:-500}"

echo "✅ difficulty curriculum complete"

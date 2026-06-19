#!/usr/bin/env bash
# Curriculum training for procedurally-generated Crystal Caves.
#
# The DQN learns far faster when it masters the easiest level family before the
# harder ones, instead of facing all four cold. Each stage trains on a wider set
# of families and warm-starts from the previous stage's saved model (the trainer
# auto-loads models/crystal_caves/crystal_caves_final.pth on launch).
#
#   EP=400 scripts/train_crystal_curriculum.sh        # episodes per stage
#   FRESH=1 ...                                        # wipe the model first
#
# PYTHONUNBUFFERED keeps the per-episode log streaming (so progress is visible
# and not lost if the run is time-boxed).
set -euo pipefail
cd "$(dirname "$0")/.."

EP="${EP:-400}"
COMMON=(--headless --game crystal_caves --random-caves --turbo --vec-envs 8 --episodes "$EP")
export PYTHONUNBUFFERED=1

if [[ "${FRESH:-1}" == "1" ]]; then
  echo "🧹 fresh start: clearing models/crystal_caves/*.pth"
  rm -f models/crystal_caves/*.pth
fi

stages=(
  "platform_network"
  "platform_network,snake_bands"
  "platform_network,snake_bands,terrain_climb"
  "platform_network,snake_bands,terrain_climb,corridor_maze"
)

for i in "${!stages[@]}"; do
  fams="${stages[$i]}"
  echo ""
  echo "================ CURRICULUM STAGE $((i + 1))/${#stages[@]}: ${fams} ================"
  python main.py "${COMMON[@]}" --cave-families "$fams"
done

echo "✅ curriculum complete"

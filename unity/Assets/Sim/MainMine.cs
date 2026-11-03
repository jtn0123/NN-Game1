#nullable enable
using System;
using System.Collections.Generic;

namespace CrystalCaves.Sim
{
    /// <summary>The connecting mine: shared movement and shooting, no cave objectives (Python MainMine).</summary>
    internal sealed class MainMine : CaveEngine
    {
        public int PortalLevel = -1;

        public MainMine(List<string> sounds) : base(new[] { CaveData.Mine }, sounds) { }

        public int NearEntrance()
        {
            var player = PlayerRect();
            for (var level = 0; level < CaveData.MineEntrances.Length; level++)
            {
                var (col, row) = CaveData.MineEntrances[level];
                if (Math.Abs(player.CenterX - (col * 32 + 16)) <= 22 && Math.Abs(player.Bottom - (row + 1) * 32) <= 8)
                    return level;
            }
            return -1;
        }

        public void Step(HumanControls controls)
        {
            Steps++;
            UpdateVisualEvents();
            ShootCooldown = Math.Max(0, ShootCooldown - 1);
            ApplyPlayerInput(controls);
            if (IsOnLadder() && !controls.Jump && !controls.Interact) Vy = 0.0; // Hold a chain until Up/Down.
            if (controls.Shoot)
            {
                if (AmmoCount <= 0 && ShootCooldown == 0)
                {
                    Play("empty");
                    ShootCooldown = ShootCooldownFrames;
                }
                else TryShoot();
            }
            MovePlayer();
            UpdateBullets();
            PortalLevel = controls.Interact ? NearEntrance() : -1;
        }
    }
}

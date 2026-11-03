#nullable enable
using System;
using System.Collections.Generic;

namespace CrystalCaves.Sim
{
    /// <summary>
    /// A cave with the classic human-play rules (Python ClassicCaves: HumanControlsMixin,
    /// ClassicSecrets, ClassicMechanisms, ClassicFlyers over CrystalCaves).
    /// </summary>
    internal sealed class ClassicCave : CaveEngine
    {
        public readonly List<BatEgg> BatEggs = new List<BatEgg>();
        public readonly List<GreenThorn> Thorns = new List<GreenThorn>();
        public readonly List<Stalactite> Stalactites = new List<Stalactite>();
        public readonly List<SecretSite> SecretCaches = new List<SecretSite>();
        public readonly HashSet<(int Col, int Row)> HiddenCrystals = new HashSet<(int, int)>();
        int nextEgg;

        public ClassicCave(IReadOnlyList<CaveSpec> caves, List<string> sounds) : base(caves, sounds) { }

        protected override void LoadLevel(CaveSpec level)
        {
            base.LoadLevel(level);
            SetupSecrets(level);
            BatEggs.Clear();
            nextEgg = 0;
            Thorns.Clear();
            Stalactites.Clear();
            for (var row = 0; row < level.Layout.Count; row++)
            {
                var line = level.Layout[row];
                for (var col = 0; col < line.Length; col++)
                {
                    if (line[col] == 't') Thorns.Add(new GreenThorn { Col = col, Row = row });
                    if (line[col] == 'v') Stalactites.Add(new Stalactite { Col = col, Row = row, Y = row * TileSize });
                }
            }
            // Lower approach trap first, for deterministic replay and capture ids.
            Stalactites.Sort((a, b) => a.Row != b.Row ? b.Row.CompareTo(a.Row) : a.Col.CompareTo(b.Col));
            // The reference scene's floor thorn sits directly beneath a hover lift.
            if (LevelIndex == 2 && Elevators.Count > 0)
                Thorns.Add(new GreenThorn { Col = Elevators[0].Col, Row = Elevators[0].Bottom });

            int ground = 0, flying = 0;
            foreach (var enemy in Enemies)
            {
                string appearance;
                if (enemy.Kind == EnemyKind.Flyer)
                {
                    appearance = (flying + LevelIndex) % 2 == 0 ? Appearance.Bat : Appearance.Eye;
                    flying++;
                }
                else
                {
                    appearance = ((ground + LevelIndex) % 3) switch
                    {
                        0 => Appearance.Slug,
                        1 => Appearance.Rock,
                        _ => Appearance.Dinosaur,
                    };
                    ground++;
                    if (appearance == Appearance.Dinosaur && !TallPatrolFits(level, enemy)) appearance = Appearance.Rock;
                }
                // Spawned with the 24px base body; tall creatures grow upward from the floor.
                enemy.Vx = 68.0 / 60.0;
                enemy.Appearance = appearance;
                enemy.Health = appearance == Appearance.Dinosaur ? 5 : 1;
                enemy.Asleep = appearance == Appearance.Rock;
                enemy.Y -= enemy.Height - 24;
            }
        }

        /// <summary>The whole connected floor must have headroom for the tall creature.</summary>
        static bool TallPatrolFits(CaveSpec level, Enemy enemy)
        {
            var layout = level.Layout;
            var col = PyMath.FloorDiv32(enemy.X + Enemy.Width / 2.0);
            var foot = PyMath.FloorDiv32(enemy.Y + 24);
            var top = PyMath.FloorDiv32(enemy.Y + 24 - 64);
            if (top < 0 || foot >= layout.Count) return false;
            bool Blocking(char c) => c == '#' || c == 'D' || c == 'd';
            bool Corridor(int c) =>
                0 <= c && c < layout[foot].Length && layout[foot][c] == '#' && !Blocking(layout[foot - 1][c]);
            int left = col, right = col;
            while (Corridor(left - 1)) left--;
            while (Corridor(right + 1)) right++;
            if (!Corridor(col)) return false;
            for (var r = top; r < foot; r++)
                for (var c = left; c <= right; c++)
                    if (Blocking(layout[r][c])) return false;
            return true;
        }

        void SetupSecrets(CaveSpec level)
        {
            SecretCaches.Clear();
            HiddenCrystals.Clear();
            foreach (var site in CaveData.Secrets)
                if (site.Cave == level.Name) SecretCaches.Add(site);
            foreach (var cache in SecretCaches)
            {
                HiddenCrystals.Add(cache.Crystal);
                if (level.At(cache.Block.Col, cache.Block.Row) != '#' || !Crystals.Contains(cache.Crystal))
                    throw new InvalidOperationException($"invalid secret cache in {level.Name}");
            }
        }

        // ---- step --------------------------------------------------------------------

        /// <summary>Python CrystalCaves.step with the classic overrides, for one human frame.</summary>
        public void Step(HumanControls controls)
        {
            if (GameOver) return;
            Steps++;
            InvulnTimer = Math.Max(0, InvulnTimer - 1);
            ShakeTimer = Math.Max(0, ShakeTimer - 1);
            ShootCooldown = Math.Max(0, ShootCooldown - 1);
            SuperTimer = Math.Max(0, SuperTimer - 1);
            FreezeTimer = Math.Max(0, FreezeTimer - 1);
            UpdateVisualEvents();
            if (GravityTimer > 0)
            {
                GravityTimer--;
                if (GravityTimer == 0) GravityDir = 1;
            }
            UpdateElevators();
            ApplyPlayerInput(controls);
            if (controls.Shoot) TryShoot();
            MovePlayer();
            if (controls.Interact) TryInteract();
            CollectPickups();
            UpdateBullets();
            UpdateEnemies();
            // Exit before danger: reaching the open exit wins a same-frame fatal hit.
            CheckExit();
            CheckPlayerDanger();
        }

        /// <summary>A lift shaft is not an invisible ladder; jump leaves the platform.</summary>
        public override bool IsOnLadder()
        {
            var (left, right, top, bottom) = TileBounds(PlayerRect());
            for (var row = top; row <= bottom; row++)
                for (var col = left; col <= right; col++)
                    if (Ladders.Contains((col, row))) return true;
            return false;
        }

        void UpdateElevators()
        {
            for (var index = 0; index < Elevators.Count; index++)
            {
                var lift = Elevators[index];
                var oldPos = lift.Pos;
                var oldDirection = lift.Direction;
                var oldRect = ElevatorSolid[index];
                var player = PlayerRect();
                var overlaps = player.Right > oldRect.Left && player.Left < oldRect.Right;
                var riding = overlaps && Vy * GravityDir >= 0
                    && (GravityDir > 0 ? player.Bottom == oldRect.Top : player.Top == oldRect.Bottom);
                lift.Pos += 68.0 / 60.0 / 32.0 * lift.Direction;
                if (lift.Pos >= lift.Bottom) { lift.Pos = lift.Bottom; lift.Direction = -1; }
                else if (lift.Pos <= lift.Top) { lift.Pos = lift.Top; lift.Direction = 1; }
                RefreshElevatorRects();
                var rect = ElevatorSolid[index];
                // Also catch a rising pad that meets a player waiting at the shaft base.
                var caught = player.Collides(rect) && !player.Collides(oldRect);
                if (!riding && !caught) continue;
                var above = riding ? GravityDir > 0 : player.CenterY < oldRect.CenterY;
                var target = above ? rect.Top - PlayerHeight : rect.Bottom;
                var others = new List<IntRect>(ElevatorSolid);
                others.RemoveAt(index);
                ElevatorSolid = others;
                var start = PlayerY;
                MoveAxis(0.0, target - start);
                if (Math.Abs(PlayerY - target) > 0.01)
                {
                    // A low ceiling stops and reverses the pad; it cannot shove Mylo into stone.
                    PlayerY = start;
                    lift.Pos = oldPos;
                    lift.Direction = -oldDirection;
                }
                else Vy = 0.0;
                RefreshElevatorRects();
            }
        }

        protected override void TryShoot()
        {
            if (AmmoCount <= 0 && ShootCooldown == 0)
            {
                ShootCooldown = ShootCooldownFrames;
                Play("empty");
            }
            base.TryShoot();
        }

        /// <summary>Head-bump secret reveal after an upward move (Python ClassicSecrets._move_axis).</summary>
        protected override void MoveAxis(double dx, double dy)
        {
            var oldY = PlayerY;
            var oldVy = Vy;
            base.MoveAxis(dx, dy);
            if (!(dy < 0 && oldVy < 0 && GravityDir == 1 && Vy == 0 && PlayerY - oldY > dy + 0.001)) return;
            var touched = PlayerRect(PlayerX, PlayerY - 1);
            foreach (var cache in SecretCaches)
            {
                if (Touches(touched, cache.Block) && HiddenCrystals.Contains(cache.Crystal))
                {
                    HiddenCrystals.Remove(cache.Crystal);
                    AddTileEvent(cache.Block, "sparkle", "SECRET", 45);
                    Play("pickup");
                }
            }
        }

        /// <summary>
        /// Python iterates a set and stops at the first lever in reach. Authored caves never
        /// place two levers within reach of one tile (enforced by tests/test_sim_trace.py),
        /// so the sorted order here selects the same lever.
        /// </summary>
        void TryInteract()
        {
            var (playerCol, playerRow) = PlayerTileOf();
            foreach (var lever in Sorted(Switches))
            {
                if (Math.Abs(lever.Col - playerCol) > 1 || Math.Abs(lever.Row - playerRow) > 1) continue;
                if (UsedSwitches.Add(lever))
                {
                    var color = SwitchColor.TryGetValue(lever, out var value) ? value : "red";
                    OpenColors.Add(color);
                    Score += 250;
                    AddTileEvent(lever, "score", "+250", 34, -10);
                    foreach (var door in Sorted(Doors))
                        if ((DoorColor.TryGetValue(door, out var doorColor) ? doorColor : "red") == color)
                            AddTileEvent(door, "sparkle", "OPEN", 42);
                    Play("switch");
                }
                break;
            }
        }

        void CollectPickups()
        {
            var touched = PlayerRect();
            foreach (var tile in Sorted(Crystals))
            {
                if (!Touches(touched, tile) || HiddenCrystals.Contains(tile)) continue;
                Crystals.Remove(tile);
                Score += 100;
                AddTileEvent(tile, "sparkle", "+100", 34);
                Play("gem");
            }
            // Hidden gems stay in the set, so the last visible gem cannot open the exit early.
            if (Crystals.Count == 0 && !AllCrystalsBonusGiven)
            {
                AllCrystalsBonusGiven = true;
                ExitUnlocked = true;
                Score += 500;
                AddTileEvent(ExitPos, "sparkle", "EXIT OPEN", 58);
                Play("win");
            }
            foreach (var tile in Sorted(AmmoPickups))
            {
                if (!Touches(touched, tile) || HiddenCrystals.Contains(tile)) continue;
                AmmoPickups.Remove(tile);
                AmmoCount += 5;
                Score += 75;
                AddTileEvent(tile, "score", "AMMO", 30);
                Play("pickup");
            }
            foreach (var tile in Sorted(Treasures))
            {
                if (!Touches(touched, tile) || HiddenCrystals.Contains(tile)) continue;
                Treasures.Remove(tile);
                Score += 300;
                AddTileEvent(tile, "sparkle", "+300", 38);
            }
            foreach (var (tile, power) in Powerups.ToArray())
            {
                if (!Touches(touched, tile) || HiddenCrystals.Contains(tile)) continue;
                Powerups.Remove((tile, power));
                Score += 125;
                AddTileEvent(tile, "sparkle", char.ToUpperInvariant(power).ToString(), 42);
                Play("pickup");
                if (power == PowerShot) SuperTimer = MaxPowerTimer;
                else if (power == GravityPower)
                {
                    GravityDir *= -1;
                    GravityTimer = 360;
                    Vy = 0.0;
                    Play("gravity");
                }
                else if (power == FreezePower) FreezeTimer = 300;
            }
        }

        void CheckExit()
        {
            if (GameOver || !ExitUnlocked) return;
            if (!PlayerRect().Collides(TileRect(ExitPos).Inflate(-6, -2))) return;
            WonLevelIndex = LevelIndex;
            GameOver = true;
            Won = true;
            EndReason = "won";
            Score += 1000 + Health * 250 + AmmoCount * 10;
            LevelIndex = (LevelIndex + 1) % CaveCount;
            Play("door");
        }

        // ---- creatures ---------------------------------------------------------------

        bool SeesPlayer(Enemy enemy, bool forwardOnly = false)
        {
            var player = PlayerRect();
            var body = enemy.Rect;
            if (player.Bottom <= body.Top || player.Top >= body.Bottom) return false;
            var dx = player.CenterX - body.CenterX;
            if (forwardOnly && dx * enemy.Vx < 0) return false;
            // Walls and closed gates block awareness at the player's mid-body height.
            var row = PyMath.FloorDiv(player.CenterY, TileSize);
            var a = PyMath.FloorDiv(player.CenterX, 32);
            var b = PyMath.FloorDiv(body.CenterX, 32);
            var start = Math.Min(a, b);
            var end = Math.Max(a, b);
            for (var col = start + 1; col < end; col++)
                if (SolidAt(col, row)) return false;
            return true;
        }

        void UpdateEnemies()
        {
            foreach (var bullet in Bullets.ToArray())
            {
                foreach (var enemy in Enemies)
                {
                    if (!enemy.Alive || !bullet.Rect.Collides(enemy.Rect)) continue;
                    Bullets.Remove(bullet);
                    if (enemy.Appearance == Appearance.Rock && !bullet.Powered)
                    {
                        AddVisualEvent("spark", bullet.X, bullet.Y, 10);
                        break;
                    }
                    enemy.Health = bullet.Powered ? 0 : enemy.Health - 1;
                    if (enemy.Health > 0)
                    {
                        enemy.HitUntil = Steps + 30;
                        enemy.Charging = true;
                        enemy.Vx = Math.Abs(enemy.Vx) * (PlayerX > enemy.X ? 1 : -1);
                        Play("enemy_hurt");
                        AddVisualEvent("spark", bullet.X, bullet.Y, 10);
                    }
                    else
                    {
                        enemy.Alive = false;
                        var points = bullet.Powered ? 250 : 200;
                        Score += points;
                        var x = enemy.X + Enemy.Width / 2.0;
                        var y = enemy.Y + enemy.Height / 2.0;
                        var facing = bullet.Vx >= 0 ? 1 : -1;
                        if (enemy.Appearance == Appearance.Dinosaur) AddVisualEvent("bones", x, y, 72, $"+{points}", facing);
                        else if (enemy.Appearance == Appearance.Eye) AddVisualEvent("slime_pulse", x, y, 36, $"+{points}", facing);
                        else AddVisualEvent("poof", x, y, 36, $"+{points}");
                    }
                    break;
                }
            }

            if (FreezeTimer > 0) return;
            for (var index = 0; index < Enemies.Count; index++)
            {
                var enemy = Enemies[index];
                if (!enemy.Alive) continue;
                if (enemy.Asleep)
                {
                    if (SeesPlayer(enemy))
                    {
                        enemy.Asleep = false;
                        enemy.WakeTimer = 14;
                        enemy.Vx = Math.Abs(enemy.Vx) * (PlayerX > enemy.X ? 1 : -1);
                    }
                    continue;
                }
                if (enemy.WakeTimer != 0)
                {
                    enemy.WakeTimer--;
                    continue;
                }
                if (enemy.Appearance == Appearance.Dinosaur)
                {
                    enemy.Charging = enemy.Charging || SeesPlayer(enemy, true);
                    enemy.Vx = (enemy.Charging ? 136 : 68) / 60.0 * (enemy.Vx > 0 ? 1 : -1);
                }
                if (enemy.Kind == EnemyKind.Flyer) AdvanceFlyer(enemy, index);
                else
                {
                    enemy.X += enemy.Vx;
                    var ahead = enemy.X + (enemy.Vx > 0 ? Enemy.Width + 2 : -2);
                    var foot = enemy.Y + enemy.Height + 2;
                    if (RectCollidesSolid(enemy.Rect) || !SolidAt(PyMath.FloorDiv32(ahead), PyMath.FloorDiv32(foot)))
                    {
                        enemy.X -= enemy.Vx;
                        enemy.Vx *= -1;
                    }
                }
            }
        }

        void AdvanceFlyer(Enemy enemy, int index)
        {
            if (enemy.Appearance == Appearance.Bat)
            {
                var turn = 60 + (index * 137 + LevelIndex * 73) % 540;
                if (Steps > 0 && Steps % turn == 0) enemy.Vx *= -1;
            }
            enemy.X += enemy.Vx;
            if (RectCollidesSolid(enemy.Rect))
            {
                enemy.X -= enemy.Vx;
                enemy.Vx *= -1;
            }
            if (enemy.Appearance == Appearance.Eye)
            {
                // A two-axis room patrol, not a sine-wave animation or a chase.
                var roam = enemy.RoamVy ?? 0.75 * ((index + LevelIndex) % 2 != 0 ? 1 : -1);
                var interval = 150 + (index * 47 + LevelIndex * 29) % 180;
                if (Steps > 0 && Steps % interval == 0) roam *= -1;
                enemy.Y += roam;
                if (RectCollidesSolid(enemy.Rect))
                {
                    enemy.Y -= roam;
                    roam *= -1;
                }
                enemy.RoamVy = roam;
            }
            else if (enemy.Appearance == Appearance.Bat)
            {
                var interval = 180 + (index * 31 + LevelIndex * 17) % 120;
                if (Steps > 0 && Steps % interval == 0 && BatEggs.Count < 8)
                {
                    var egg = new BatEgg { Identity = nextEgg, X = enemy.X + 6, Y = enemy.Y + enemy.Height };
                    if (!RectCollidesSolid(egg.Rect))
                    {
                        BatEggs.Add(egg);
                        nextEgg++;
                    }
                }
            }
        }

        // ---- hazards -----------------------------------------------------------------

        void CheckPlayerDanger()
        {
            if (GameOver) return;
            UpdateThorns();
            var fallingHit = UpdateStalactites();
            var eggHit = UpdateBatEggs();
            var player = PlayerRect();
            var hazard = false;
            foreach (var tile in Hazards) if (player.Collides(TileRect(tile))) { hazard = true; break; }
            if (!hazard) foreach (var thorn in Thorns) if (player.Collides(thorn.Rect)) { hazard = true; break; }
            hazard = hazard || fallingHit || eggHit;
            var touchedEnemy = false;
            foreach (var enemy in Enemies) if (enemy.Alive && player.Collides(enemy.Rect)) { touchedEnemy = true; break; }
            if (hazard || touchedEnemy) DamagePlayer();
        }

        void UpdateThorns()
        {
            var player = PlayerRect();
            foreach (var thorn in Thorns)
            {
                var top = thorn.Row;
                while (top > 0 && !SolidAt(thorn.Col, top - 1)) top--;
                var detection = new IntRect(thorn.Col * 32, top * 32, 32, (thorn.Row - top + 1) * 32);
                if (!player.Collides(detection))
                {
                    thorn.Frame = thorn.Clock = 0;
                    thorn.Triggered = false;
                    continue;
                }
                if (!thorn.Triggered)
                {
                    Play("thorn");
                    thorn.Triggered = true;
                }
                // Discrete original-style poses at 17 Hz, independent of rendering.
                thorn.Clock += 17;
                if (thorn.Clock >= 60)
                {
                    thorn.Frame = Math.Min(4, thorn.Frame + 1);
                    thorn.Clock -= 60;
                }
            }
        }

        bool UpdateStalactites()
        {
            var player = PlayerRect();
            var hit = false;
            foreach (var trap in Stalactites)
            {
                if (!trap.Alive) continue;
                hit = hit || player.Collides(trap.Rect);
                if (!trap.Falling)
                {
                    var bottom = trap.Row + 1;
                    while (bottom < LevelRows && !SolidAt(trap.Col, bottom)) bottom++;
                    var detection = new IntRect(trap.Col * 32, trap.Y, 32, bottom * 32 - trap.Y);
                    if (!player.Collides(detection)) continue;
                    trap.Falling = true;
                    Play("stalactite");
                }
                trap.Clock += 17;
                if (trap.Clock < 60) continue;
                trap.Clock -= 60;
                // Sweep the narrow tip so fast motion cannot tunnel through Mylo or a gate.
                for (var pixel = 0; pixel < 16; pixel++)
                {
                    trap.Y += 1;
                    if (RectCollidesSolid(trap.Rect) || trap.Y >= LevelHeight)
                    {
                        trap.Alive = false;
                        break;
                    }
                    hit = hit || player.Collides(trap.Rect);
                }
            }
            return hit;
        }

        bool UpdateBatEggs()
        {
            var player = PlayerRect();
            var hit = false;
            var remaining = new List<BatEgg>(BatEggs.Count);
            foreach (var egg in BatEggs)
            {
                Bullet? struck = null;
                foreach (var bullet in Bullets) if (bullet.Rect.Collides(egg.Rect)) { struck = bullet; break; }
                if (struck != null)
                {
                    Bullets.Remove(struck);
                    AddVisualEvent("spark", egg.X + 6, egg.Y + 8, 10);
                    continue;
                }
                if (FreezeTimer > 0)
                {
                    hit = hit || player.Collides(egg.Rect);
                    remaining.Add(egg);
                    continue;
                }
                egg.Ttl -= 1;
                egg.Vy = Math.Min(4.0, egg.Vy + 0.12);
                // Sweep each pixel: the egg cannot tunnel through floors or Mylo.
                var landed = false;
                var distance = egg.Vy;
                while (distance > 0)
                {
                    var step = Math.Min(1.0, distance);
                    egg.Y += step;
                    distance -= step;
                    if (RectCollidesSolid(egg.Rect))
                    {
                        landed = true;
                        AddVisualEvent("poof", egg.X + 6, egg.Y + 8, 12);
                        break;
                    }
                    if (player.Collides(egg.Rect))
                    {
                        hit = landed = true;
                        break;
                    }
                }
                if (!landed && egg.Ttl > 0 && egg.Y < LevelHeight) remaining.Add(egg);
            }
            BatEggs.Clear();
            BatEggs.AddRange(remaining);
            return hit;
        }
    }
}

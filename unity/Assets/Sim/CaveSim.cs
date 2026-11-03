#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("CrystalCaves.SimTests")]

namespace CrystalCaves.Sim
{
    /// <summary>
    /// The standalone human-play session: one cave game, the connecting mine and the
    /// presentation snapshot. A port of Python <c>CaveSession</c> for human mode,
    /// verified frame by frame against recorded reference traces.
    /// </summary>
    public sealed class CaveSim
    {
        public const int MaxBatch = 8;
        public const int CaveCount = 16;

        readonly List<string> sounds = new List<string>();
        readonly ClassicCave cave;
        readonly string[] caveNames;
        MainMine? mine;
        CaveEngine game;
        readonly HashSet<int> cleared = new HashSet<int>();
        readonly HashSet<VisualEvent> muzzleEffects = new HashSet<VisualEvent>(ReferenceComparer.Instance);
        string[] enemySprites = Array.Empty<string>();
        bool humanOnly = true;
        int episode;

        public CaveSim(int level = 0)
        {
            if (CaveData.Caves.Length != CaveCount) throw new InvalidOperationException("cave data must hold sixteen caves");
            cave = new ClassicCave(CaveData.Caves, sounds);
            game = cave;
            caveNames = Array.ConvertAll(CaveData.Caves, spec => spec.Name);
            Reset(level, announce: false);
        }

        public string Realm { get; private set; } = "cave";
        public bool InMine => Realm == "mine";

        /// <summary>Begin a cave expedition (protocol <c>reset</c>).</summary>
        public SimSnapshot Reset(int level) => Reset(level, announce: true);

        SimSnapshot Reset(int level, bool announce)
        {
            if (level < 0 || level >= CaveCount)
                throw new ArgumentOutOfRangeException(nameof(level), $"level must be an integer from 0 to {CaveCount - 1}");
            game = cave;
            Realm = "cave";
            cave.EvalCursor = level;
            cave.Reset();
            enemySprites = cave.Enemies.ConvertAll(enemy => enemy.Appearance).ToArray();
            sounds.Clear();
            muzzleEffects.Clear();
            episode++;
            humanOnly = true;
            if (announce) sounds.Add("enter");
            return Snapshot(clearSounds: false);
        }

        /// <summary>Enter the connecting mine, keeping Mylo's last mine position (protocol <c>mine</c>).</summary>
        public SimSnapshot OpenMine(IEnumerable<int> clearedCaves)
        {
            if (clearedCaves == null) throw new ArgumentNullException(nameof(clearedCaves));
            var completed = new HashSet<int>();
            foreach (var level in clearedCaves)
            {
                if (level < 0 || level >= CaveCount)
                    throw new ArgumentOutOfRangeException(nameof(clearedCaves), $"cleared cave must be an integer from 0 to {CaveCount - 1}");
                completed.Add(level);
            }
            var returning = mine != null;
            mine ??= new MainMine(sounds);
            cleared.Clear();
            cleared.UnionWith(completed);
            var (x, y, facing) = (mine.PlayerX, mine.PlayerY, mine.Facing);
            game = mine;
            mine.EvalCursor = 0;
            mine.Reset();
            if (returning) (mine.PlayerX, mine.PlayerY, mine.Facing) = (x, y, facing);
            mine.PortalLevel = -1;
            Realm = "mine";
            humanOnly = false;
            enemySprites = Array.Empty<string>();
            sounds.Clear();
            muzzleEffects.Clear();
            episode++;
            return Snapshot(clearSounds: false);
        }

        /// <summary>Advance up to <see cref="MaxBatch"/> frames of human buttons (protocol <c>human_step</c>).</summary>
        public SimSnapshot HumanStep(IReadOnlyList<HumanControls> controls)
        {
            if (controls == null || controls.Count < 1 || controls.Count > MaxBatch)
                throw new ArgumentException($"controls must contain 1 to {MaxBatch} controls", nameof(controls));
            sounds.Clear();
            foreach (var control in controls)
            {
                if (game.GameOver || (mine != null && game == mine && mine.PortalLevel >= 0)) break;
                StepFrame(control);
            }
            return Snapshot(clearSounds: false);
        }

        /// <summary>Report the current state without advancing (protocol <c>snapshot</c>).</summary>
        public SimSnapshot Snapshot() => Snapshot(clearSounds: true);

        void StepFrame(HumanControls control)
        {
            var beforeAmmo = game.AmmoPickups.Count;
            var beforeTreasure = game.Treasures.Count;
            var beforeEnemies = AliveEnemies();
            var beforePower = game.Powerups.ToArray();
            var beforeHealth = game.Health;
            var powered = game.SuperTimer > 0;
            var eventStart = sounds.Count;
            // Hold the previous objects so an expired muzzle cannot be confused with a new one.
            var previousEffects = new HashSet<VisualEvent>(game.VisualEvents, ReferenceComparer.Instance);

            if (game == cave) cave.Step(control);
            else mine!.Step(control);

            if (sounds.IndexOf("shoot", eventStart) >= 0)
            {
                // Hide only the newborn ten-step muzzle spark; impacts stay visible.
                foreach (var visual in game.VisualEvents)
                {
                    if (visual.Kind == "spark" && visual.MaxTtl == 10 && !previousEffects.Contains(visual))
                    {
                        muzzleEffects.Add(visual);
                        break;
                    }
                }
            }
            muzzleEffects.IntersectWith(game.VisualEvents);
            VoiceSounds(eventStart, beforeAmmo, beforeTreasure, beforeEnemies, beforePower, beforeHealth, powered);
        }

        int AliveEnemies()
        {
            var alive = 0;
            foreach (var enemy in game.Enemies) if (enemy.Alive) alive++;
            return alive;
        }

        /// <summary>Name the engine's sound events for the classic sound bank, without changing state.</summary>
        void VoiceSounds(int start, int ammo, int treasure, int enemies,
            ((int Col, int Row) Tile, char Kind)[] powerups, int health, bool powered)
        {
            var pickups = new Queue<string>();
            for (var index = 0; index < ammo - game.AmmoPickups.Count; index++) pickups.Enqueue("ammo");
            foreach (var (tile, kind) in powerups)
            {
                var remains = false;
                foreach (var (other, _) in game.Powerups) if (other == tile) { remains = true; break; }
                if (!remains) pickups.Enqueue(kind == CaveEngine.FreezePower ? "freeze" : "pickup");
            }
            var voiced = new List<string>();
            for (var index = start; index < sounds.Count; index++)
            {
                var name = sounds[index];
                if (name == "pickup" && pickups.Count > 0) name = pickups.Dequeue();
                if (name == "shoot" && powered) name = "power_shoot";
                if (name == "win" && !game.Won) continue; // The exit owns the fanfare.
                if (name == "door" && game.Won) name = "win";
                voiced.Add(name);
            }
            if (game.Treasures.Count < treasure) voiced.Add("treasure");
            if (AliveEnemies() < enemies && game.Health == health) voiced.Add("enemy_defeat");
            sounds.RemoveRange(start, sounds.Count - start);
            sounds.AddRange(voiced);
        }

        // ---- snapshot ----------------------------------------------------------------

        SimSnapshot Snapshot(bool clearSounds)
        {
            if (clearSounds) sounds.Clear();
            var snapshot = new SimSnapshot
            {
                Realm = Realm,
                ClearedCaves = cleared.Count,
                NearEntrance = InMine ? mine!.NearEntrance() : -1,
                PortalLevel = InMine ? mine!.PortalLevel : -1,
                Episode = episode,
                Level = InMine ? -1 : game.Won ? game.WonLevelIndex : game.LevelIndex,
                LevelName = game.Level.Name,
                Levels = caveNames,
                Layout = TerrainLayout(),
                Cols = game.LevelCols,
                Rows = game.LevelRows,
                Health = game.Health,
                Ammo = game.AmmoCount,
                Score = game.Score,
                Crystals = game.Crystals.Count,
                InitialCrystals = game.InitialCrystals,
                ExitUnlocked = game.ExitUnlocked,
                Steps = game.Steps,
                FreezeTimer = game.FreezeTimer,
                SuperTimer = game.SuperTimer,
                Done = game.GameOver,
                Won = game.Won,
                EndReason = game.EndReason,
                Mode = "human",
                HumanOnly = humanOnly,
                Sounds = sounds.ToArray(),
            };
            snapshot.Player = PlayerSnapshot(out var sprite);
            if (InMine) AddMineEntities(snapshot.Entities, sprite);
            else AddCaveEntities(snapshot.Entities, sprite);
            foreach (var visual in game.VisualEvents)
            {
                if (muzzleEffects.Contains(visual)) continue;
                snapshot.Effects.Add(new SimEffect
                {
                    Id = string.Format(CultureInfo.InvariantCulture, "{0}_{1:R}_{2:R}_{3}",
                        visual.Kind, visual.X, visual.Y, game.Steps - visual.MaxTtl + visual.Ttl),
                    Kind = visual.Kind, X = visual.X, Y = visual.Y,
                    Ttl = visual.Ttl, MaxTtl = visual.MaxTtl, Text = visual.Text, Facing = visual.Facing,
                });
            }
            return snapshot;
        }

        string[] TerrainLayout()
        {
            var rows = new string[game.LevelRows];
            for (var row = 0; row < game.LevelRows; row++)
            {
                var line = (char[])game.Grid[row].Clone();
                foreach (var pair in game.HazardKinds) if (pair.Key.Row == row) line[pair.Key.Col] = pair.Value;
                rows[row] = new string(line);
            }
            return rows;
        }

        SimPlayer PlayerSnapshot(out string sprite)
        {
            var climbing = game.IsOnLadder() && Math.Abs(game.Vy) > 0.1;
            if (game.ShootCooldown > CaveEngine.ShootCooldownFrames - 7)
                sprite = !game.Grounded && !climbing ? "mylo_shoot_air" : "mylo_shoot";
            else if (!game.Grounded) sprite = "mylo_jump";
            else if (Math.Abs(game.Vx) > 0.2) sprite = game.Steps / 8 % 2 == 0 ? "mylo_walk_1" : "mylo_walk_2";
            else sprite = "mylo_idle";
            return new SimPlayer
            {
                X = game.PlayerX, Y = game.PlayerY, Vx = game.Vx, Vy = game.Vy, Sprite = sprite,
                Facing = game.Facing, Grounded = game.Grounded, GravityDir = game.GravityDir,
                Invulnerable = game.InvulnTimer > 0, InvulnerabilityLeft = game.InvulnTimer,
                InvulnerabilityFrames = CaveEngine.InvulnFrames, Climbing = climbing,
            };
        }

        static SimEntity Entity(string id, string sprite, double x, double y, double w = 32, double h = 32,
            string glow = "", bool flip = false) =>
            new SimEntity { Id = id, Sprite = sprite, X = x, Y = y, W = w, H = h, Glow = glow, Flip = flip };

        /// <summary>Gates, exits and air machines grow into an authored empty cell above.</summary>
        int TallHeight((int Col, int Row) tile) =>
            tile.Row > 0 && game.Level.At(tile.Col, tile.Row - 1) == '.' ? 64 : 32;

        void AddCaveEntities(List<SimEntity> entities, string playerSprite)
        {
            foreach (var (col, row) in CaveEngine.Sorted(cave.Crystals))
                if (!cave.HiddenCrystals.Contains((col, row)))
                    entities.Add(Entity($"crystal_{col}_{row}", "crystal_blue", col * 32, row * 32, glow: "cyan"));
            foreach (var (col, row) in CaveEngine.Sorted(cave.AmmoPickups))
                entities.Add(Entity($"ammo_{col}_{row}", "ammo", col * 32, row * 32, glow: "amber"));
            foreach (var (col, row) in CaveEngine.Sorted(cave.Treasures))
                entities.Add(Entity($"treasure_{col}_{row}", "treasure", col * 32, row * 32, glow: "amber"));
            foreach (var secret in cave.SecretCaches)
            {
                var (col, row) = secret.Block;
                var armed = cave.HiddenCrystals.Contains(secret.Crystal);
                entities.Add(Entity($"secret_cache_{col}_{row}", armed ? "secret_cache_armed" : "secret_cache_empty", col * 32, row * 32));
            }
            foreach (var tile in CaveEngine.Sorted(cave.AirTanks))
            {
                var height = TallHeight(tile);
                entities.Add(Entity($"air_{tile.Col}_{tile.Row}", height == 64 ? "air_tank_tall" : "air_tank",
                    tile.Col * 32, (tile.Row + 1) * 32 - height, h: height, glow: "cyan"));
            }
            var powers = new List<((int Col, int Row) Tile, char Kind)>(cave.Powerups);
            powers.Sort((a, b) => a.Tile.Col != b.Tile.Col ? a.Tile.Col.CompareTo(b.Tile.Col) : a.Tile.Row.CompareTo(b.Tile.Row));
            foreach (var ((col, row), kind) in powers)
            {
                var sprite = kind == CaveEngine.PowerShot ? "power_shot" : kind == CaveEngine.GravityPower ? "gravity" : "freeze";
                entities.Add(Entity($"power_{col}_{row}", sprite, col * 32, row * 32, glow: "violet"));
            }
            foreach (var tile in CaveEngine.Sorted(cave.Doors))
            {
                if (cave.DoorOpen(tile)) continue;
                var color = cave.DoorColor.TryGetValue(tile, out var value) ? value : "red";
                var height = TallHeight(tile);
                entities.Add(Entity($"door_{tile.Col}_{tile.Row}", $"door_{color}" + (height == 64 ? "_tall" : ""),
                    tile.Col * 32, (tile.Row + 1) * 32 - height, h: height));
            }
            foreach (var tile in CaveEngine.Sorted(cave.Switches))
            {
                var color = cave.SwitchColor.TryGetValue(tile, out var value) ? value : "red";
                var state = cave.UsedSwitches.Contains(tile) ? "on" : "off";
                entities.Add(Entity($"switch_{tile.Col}_{tile.Row}", $"switch_{color}_{state}", tile.Col * 32, tile.Row * 32));
            }
            var exitHeight = TallHeight(cave.ExitPos);
            entities.Add(Entity("exit", (cave.ExitUnlocked ? "exit_open" : "exit_locked") + (exitHeight == 64 ? "_tall" : ""),
                cave.ExitPos.Col * 32, (cave.ExitPos.Row + 1) * 32 - exitHeight, h: exitHeight,
                glow: cave.ExitUnlocked ? "cyan" : "violet"));
            for (var index = 0; index < cave.Enemies.Count; index++)
            {
                var enemy = cave.Enemies[index];
                if (!enemy.Alive) continue;
                var entity = Entity($"enemy_{index}", enemySprites[index], enemy.X, enemy.Y, Enemy.Width, enemy.Height, flip: enemy.Vx < 0);
                entity.Extras = EntityExtras.Creature;
                entity.Asleep = enemy.Asleep;
                entity.Hit = enemy.HitUntil > cave.Steps;
                entities.Add(entity);
            }
            for (var index = 0; index < cave.Elevators.Count; index++)
            {
                var lift = cave.Elevators[index];
                var entity = Entity($"lift_{index}", "elevator", lift.Col * 32, lift.Pos * 32);
                entity.Extras = EntityExtras.Lift;
                entity.Frame = cave.Steps / 4 % 4;
                entities.Add(entity);
            }
            foreach (var egg in cave.BatEggs)
                entities.Add(Entity($"egg_{egg.Identity}", "bat_egg", egg.X, egg.Y, 12, 16));
            for (var index = 0; index < cave.Thorns.Count; index++)
            {
                var thorn = cave.Thorns[index];
                entities.Add(Entity($"thorn_{index}", $"green_thorn_{thorn.Frame}", thorn.Col * 32, thorn.Row * 32));
            }
            for (var index = 0; index < cave.Stalactites.Count; index++)
            {
                var trap = cave.Stalactites[index];
                if (trap.Alive) entities.Add(Entity($"stalactite_{index}", "stalactite", trap.Col * 32, trap.Y));
            }
            AddBulletsAndPlayer(entities, playerSprite);
        }

        void AddMineEntities(List<SimEntity> entities, string playerSprite)
        {
            AddBulletsAndPlayer(entities, playerSprite);
            for (var index = 0; index < CaveData.MineEntrances.Length; index++)
            {
                var (col, row) = CaveData.MineEntrances[index];
                entities.Add(Entity($"entrance_{index}", cleared.Contains(index) ? "mine_door_cleared" : "mine_door", col * 32, row * 32));
            }
            for (var index = 0; index < CaveData.MineTorches.Length; index++)
            {
                var (col, row) = CaveData.MineTorches[index];
                entities.Add(Entity($"torch_{index}", "mine_torch_0", col * 32, row * 32));
            }
        }

        void AddBulletsAndPlayer(List<SimEntity> entities, string playerSprite)
        {
            for (var index = 0; index < game.Bullets.Count; index++)
            {
                var bullet = game.Bullets[index];
                entities.Add(Entity($"bullet_{index}", "bullet", bullet.X, bullet.Y, 10, 4, "amber", bullet.Vx < 0));
            }
            entities.Add(Entity("player", playerSprite, game.PlayerX - 1, game.PlayerY - 2, 24, 32, flip: game.Facing < 0));
        }

        sealed class ReferenceComparer : IEqualityComparer<VisualEvent>
        {
            public static readonly ReferenceComparer Instance = new ReferenceComparer();
            public bool Equals(VisualEvent? x, VisualEvent? y) => ReferenceEquals(x, y);
            public int GetHashCode(VisualEvent obj) => RuntimeHelpers.GetHashCode(obj);
        }
    }
}

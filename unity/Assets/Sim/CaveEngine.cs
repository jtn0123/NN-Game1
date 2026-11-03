#nullable enable
using System;
using System.Collections.Generic;

namespace CrystalCaves.Sim
{
    /// <summary>
    /// Shared movement, collision, pickups and bullets of the reference engine
    /// (Python CrystalCaves with the classic human control profile). Method names
    /// follow the Python methods they port so behavior can be compared line by line.
    /// </summary>
    internal abstract class CaveEngine
    {
        public const int TileSize = 32;
        public const int PlayerWidth = 22;
        public const int PlayerHeight = 30;
        public const int MaxHealth = 3;
        public const int ShootCooldownFrames = 14;
        public const int InvulnFrames = 70;
        public const int MaxPowerTimer = 420;
        public const int ShakeFrames = 16;
        // Classic human tuning (Python playfeel.apply_classic_controls).
        public const double MoveSpeed = 140.0 / 60.0;
        public const double AirSpeed = 140.0 / 60.0;
        public const double JumpSpeed = 5.3;
        public const double Gravity = 0.17;
        public const double MaxFallSpeed = 5.3;
        public const double Friction = 0.0;
        public const double BulletSpeed = 9.0;
        public const double LadderClimbSpeed = 3.1;
        public const double LadderDescendSpeed = 2.0;

        public const char Empty = '.', Solid = '#', Crystal = '*', Exit = 'E', Door = 'D', Door2 = 'd';
        public const char Switch = 's', Switch2 = 'S', Ammo = 'A', Treasure = '$', PowerShot = 'p';
        public const char GravityPower = 'g', FreezePower = 'z', AirTank = 'O', Crawler = 'M', Flyer = 'F';
        public const char Spike = '^', Acid = '~', ElevatorTile = '=', Ladder = 'H', PlayerTile = 'P';

        readonly IReadOnlyList<CaveSpec> caves;
        readonly List<string> sounds;

        protected CaveEngine(IReadOnlyList<CaveSpec> caves, List<string> sounds)
        {
            if (caves.Count == 0) throw new ArgumentException("at least one cave is required", nameof(caves));
            this.caves = caves;
            this.sounds = sounds;
            Level = caves[0];
            Grid = Array.Empty<char[]>();
        }

        public int CaveCount => caves.Count;
        public CaveSpec Level { get; private set; }
        public char[][] Grid { get; private set; }
        public int LevelCols, LevelRows, LevelWidth, LevelHeight;
        public int LevelIndex, WonLevelIndex, EvalCursor;

        public double PlayerX, PlayerY, Vx, Vy;
        public int Facing = 1;
        public bool Grounded;
        public int CoyoteTimer;
        public int Health = MaxHealth, AmmoCount = 5, Score, Steps;
        public bool GameOver, Won;
        public string EndReason = "running";
        public int InvulnTimer, ShootCooldown, ShakeTimer, SuperTimer, FreezeTimer, GravityTimer;
        public int GravityDir = 1;
        public readonly HashSet<string> OpenColors = new HashSet<string>();
        public bool ExitUnlocked;
        protected bool AllCrystalsBonusGiven;

        public readonly HashSet<(int Col, int Row)> Crystals = new HashSet<(int, int)>();
        public int InitialCrystals;
        public (int Col, int Row) ExitPos;
        public readonly HashSet<(int Col, int Row)> Doors = new HashSet<(int, int)>();
        public readonly HashSet<(int Col, int Row)> Switches = new HashSet<(int, int)>();
        public readonly HashSet<(int Col, int Row)> UsedSwitches = new HashSet<(int, int)>();
        public readonly Dictionary<(int Col, int Row), string> DoorColor = new Dictionary<(int, int), string>();
        public readonly Dictionary<(int Col, int Row), string> SwitchColor = new Dictionary<(int, int), string>();
        public readonly HashSet<(int Col, int Row)> Hazards = new HashSet<(int, int)>();
        public readonly Dictionary<(int Col, int Row), char> HazardKinds = new Dictionary<(int, int), char>();
        public readonly HashSet<(int Col, int Row)> AmmoPickups = new HashSet<(int, int)>();
        public readonly HashSet<(int Col, int Row)> Treasures = new HashSet<(int, int)>();
        /// <summary>Insertion (row-major) order, like the Python dict it ports.</summary>
        public readonly List<((int Col, int Row) Tile, char Kind)> Powerups = new List<((int, int), char)>();
        public readonly HashSet<(int Col, int Row)> AirTanks = new HashSet<(int, int)>();
        public readonly HashSet<(int Col, int Row)> Ladders = new HashSet<(int, int)>();
        public readonly List<Enemy> Enemies = new List<Enemy>();
        public readonly List<Elevator> Elevators = new List<Elevator>();
        public List<IntRect> ElevatorSolid = new List<IntRect>();
        public List<Bullet> Bullets = new List<Bullet>();
        public List<VisualEvent> VisualEvents = new List<VisualEvent>();

        protected void Play(string sound) => sounds.Add(sound);

        // ---- episode -----------------------------------------------------------------

        /// <summary>Python CrystalCaves.reset in evaluation mode (fixed cave cycle).</summary>
        public void Reset()
        {
            LevelIndex = EvalCursor % caves.Count;
            Level = caves[LevelIndex];
            EvalCursor++;
            LoadLevel(Level);
            Vx = Vy = 0.0;
            Facing = 1;
            Grounded = false;
            CoyoteTimer = 0;
            Health = MaxHealth;
            AmmoCount = 5;
            Score = 0;
            GameOver = Won = false;
            Steps = 0;
            InvulnTimer = ShootCooldown = ShakeTimer = SuperTimer = FreezeTimer = GravityTimer = 0;
            GravityDir = 1;
            OpenColors.Clear();
            ExitUnlocked = false;
            AllCrystalsBonusGiven = false;
            Bullets.Clear();
            VisualEvents.Clear();
            EndReason = "running";
            WonLevelIndex = LevelIndex;
        }

        protected virtual void LoadLevel(CaveSpec level)
        {
            var width = 0;
            foreach (var line in level.Layout) width = Math.Max(width, line.Length);
            var rows = level.Layout.Count;
            Grid = new char[rows][];
            for (var row = 0; row < rows; row++)
            {
                Grid[row] = new char[width];
                for (var col = 0; col < width; col++) Grid[row][col] = Empty;
            }
            LevelCols = width;
            LevelRows = rows;
            LevelWidth = LevelCols * TileSize;
            LevelHeight = LevelRows * TileSize;

            Crystals.Clear(); Doors.Clear(); Switches.Clear(); UsedSwitches.Clear();
            DoorColor.Clear(); SwitchColor.Clear(); Hazards.Clear(); HazardKinds.Clear();
            AmmoPickups.Clear(); Treasures.Clear(); Powerups.Clear(); AirTanks.Clear();
            Enemies.Clear(); Elevators.Clear(); Ladders.Clear();

            for (var row = 0; row < rows; row++)
            {
                for (var col = 0; col < width; col++)
                {
                    var tile = (col, row);
                    switch (level.At(col, row))
                    {
                        case Solid: Grid[row][col] = Solid; break;
                        case PlayerTile:
                            PlayerX = col * TileSize + 5;
                            PlayerY = row * TileSize + 1;
                            break;
                        case Crystal: Crystals.Add(tile); break;
                        // A layout without an exit keeps the previous exit (Python quirk).
                        case Exit: ExitPos = tile; break;
                        case Door: Doors.Add(tile); DoorColor[tile] = "red"; break;
                        case Door2: Doors.Add(tile); DoorColor[tile] = "blue"; break;
                        case Switch: Switches.Add(tile); SwitchColor[tile] = "red"; break;
                        case Switch2: Switches.Add(tile); SwitchColor[tile] = "blue"; break;
                        case Spike:
                        case Acid:
                            Hazards.Add(tile);
                            HazardKinds[tile] = level.At(col, row);
                            break;
                        case ElevatorTile: Grid[row][col] = ElevatorTile; break;
                        case Ladder: Grid[row][col] = Ladder; break;
                        case Ammo: AmmoPickups.Add(tile); break;
                        case Treasure: Treasures.Add(tile); break;
                        case PowerShot:
                        case GravityPower:
                        case FreezePower:
                            Powerups.Add((tile, level.At(col, row)));
                            break;
                        case AirTank: AirTanks.Add(tile); break;
                        case Crawler:
                            Enemies.Add(new Enemy { X = col * TileSize + 4, Y = row * TileSize + 8, Vx = 1.1, Kind = EnemyKind.Crawler });
                            break;
                        case Flyer:
                            Enemies.Add(new Enemy { X = col * TileSize + 4, Y = row * TileSize + 4, Vx = 1.6, Kind = EnemyKind.Flyer });
                            break;
                    }
                }
            }
            InitialCrystals = Crystals.Count;

            // Vertical runs of lift cells form shafts; each platform starts at the top.
            for (var col = 0; col < LevelCols; col++)
            {
                var row = 0;
                while (row < LevelRows)
                {
                    if (Grid[row][col] == ElevatorTile)
                    {
                        var top = row;
                        while (row < LevelRows && Grid[row][col] == ElevatorTile) row++;
                        Elevators.Add(new Elevator { Col = col, Top = top, Bottom = row - 1, Pos = top });
                    }
                    else row++;
                }
            }
            RefreshElevatorRects();
            for (var row = 0; row < LevelRows; row++)
                for (var col = 0; col < LevelCols; col++)
                    if (Grid[row][col] == Ladder) Ladders.Add((col, row));
        }

        // ---- geometry ----------------------------------------------------------------

        public IntRect PlayerRect() => PlayerRect(PlayerX, PlayerY);
        public static IntRect PlayerRect(double x, double y) =>
            new IntRect(PyMath.Trunc(x), PyMath.Trunc(y), PlayerWidth, PlayerHeight);

        public (int Col, int Row) PlayerTileOf() =>
            (PyMath.FloorDiv32(PlayerX + PlayerWidth / 2.0), PyMath.FloorDiv32(PlayerY + PlayerHeight / 2.0));

        public static (double X, double Y) TileCenter((int Col, int Row) tile) =>
            (tile.Col * TileSize + TileSize / 2.0, tile.Row * TileSize + TileSize / 2.0);

        public static IntRect TileRect((int Col, int Row) tile) =>
            new IntRect(tile.Col * TileSize, tile.Row * TileSize, TileSize, TileSize);

        /// <summary>Tile bounds a rectangle touches (Python _tiles_for_rect), inclusive.</summary>
        protected static (int Left, int Right, int Top, int Bottom) TileBounds(IntRect rect) => (
            PyMath.FloorDiv(rect.Left, TileSize),
            PyMath.FloorDiv(rect.Right - 1, TileSize),
            PyMath.FloorDiv(rect.Top, TileSize),
            PyMath.FloorDiv(rect.Bottom - 1, TileSize));

        protected static bool Touches(IntRect rect, (int Col, int Row) tile)
        {
            var (left, right, top, bottom) = TileBounds(rect);
            return left <= tile.Col && tile.Col <= right && top <= tile.Row && tile.Row <= bottom;
        }

        public bool RectCollidesSolid(IntRect rect)
        {
            var (left, right, top, bottom) = TileBounds(rect);
            for (var row = top; row <= bottom; row++)
                for (var col = left; col <= right; col++)
                    if (SolidAt(col, row)) return true;
            foreach (var platform in ElevatorSolid) if (rect.Collides(platform)) return true;
            return false;
        }

        public bool DoorOpen((int Col, int Row) tile) =>
            OpenColors.Contains(DoorColor.TryGetValue(tile, out var color) ? color : "red");

        public bool SolidAt(int col, int row)
        {
            if (col < 0 || row < 0 || col >= LevelCols || row >= LevelRows) return true;
            if (Grid[row][col] == Solid) return true;
            return Doors.Contains((col, row)) && !DoorOpen((col, row));
        }

        public bool IsOnSurface() => RectCollidesSolid(PlayerRect().Offset(0, GravityDir));

        public virtual bool IsOnLadder()
        {
            var (left, right, top, bottom) = TileBounds(PlayerRect());
            for (var row = top; row <= bottom; row++)
                for (var col = left; col <= right; col++)
                    if (0 <= col && col < LevelCols && 0 <= row && row < LevelRows
                        && (Ladders.Contains((col, row)) || Grid[row][col] == ElevatorTile))
                        return true;
            return false;
        }

        protected void RefreshElevatorRects()
        {
            var rects = new List<IntRect>(Elevators.Count);
            foreach (var lift in Elevators)
                rects.Add(new IntRect(lift.Col * TileSize, PyMath.Trunc(lift.Pos * TileSize), TileSize, TileSize));
            ElevatorSolid = rects;
        }

        // ---- player ------------------------------------------------------------------

        /// <summary>Python logic _apply_player_input plus the human ladder hold.</summary>
        protected void ApplyPlayerInput(HumanControls controls)
        {
            ApplyBaseInput(controls.Move, controls.Jump);
            if (IsOnLadder() && !(controls.Jump || controls.Interact)) Vy = 0.0;
        }

        void ApplyBaseInput(int moveDir, bool wantsJump)
        {
            if (IsOnLadder())
            {
                Grounded = false;
                CoyoteTimer = 0;
                if (moveDir != 0) { Vx = moveDir * AirSpeed; Facing = moveDir; }
                else
                {
                    Vx *= Friction;
                    if (Math.Abs(Vx) < 0.05) Vx = 0.0;
                }
                Vy = wantsJump ? -LadderClimbSpeed : LadderDescendSpeed;
                return;
            }

            Grounded = IsOnSurface();
            CoyoteTimer = Grounded ? 6 : Math.Max(0, CoyoteTimer - 1);
            var speed = Grounded ? MoveSpeed : AirSpeed;
            if (moveDir != 0) { Vx = moveDir * speed; Facing = moveDir; }
            else
            {
                Vx *= Friction;
                if (Math.Abs(Vx) < 0.05) Vx = 0.0;
            }
            if (wantsJump && CoyoteTimer > 0)
            {
                Vy = -JumpSpeed * GravityDir;
                Grounded = false;
                CoyoteTimer = 0;
                Play("jump");
            }
            Vy += Gravity * GravityDir;
            Vy = PyMath.Clip(Vy, -MaxFallSpeed, MaxFallSpeed);
        }

        protected void MovePlayer()
        {
            var wasAirborne = !Grounded;
            var fallingSpeed = Math.Abs(Vy);
            MoveAxis(Vx, 0.0);
            MoveAxis(0.0, Vy);
            Grounded = IsOnSurface();
            if (wasAirborne && Grounded && fallingSpeed > 3.0) Play("land");
        }

        /// <summary>One-pixel sweep along a single axis; stops and zeroes velocity at a wall.</summary>
        protected virtual void MoveAxis(double dx, double dy)
        {
            var remaining = dx != 0 ? dx : dy;
            if (remaining == 0) return;
            var sign = remaining > 0 ? 1.0 : -1.0;
            var horizontal = dx != 0;
            while (Math.Abs(remaining) > 0.001)
            {
                var step = sign * Math.Min(1.0, Math.Abs(remaining));
                var nextX = horizontal ? PlayerX + step : PlayerX;
                var nextY = horizontal ? PlayerY : PlayerY + step;
                if (RectCollidesSolid(PlayerRect(nextX, nextY)))
                {
                    if (horizontal) Vx = 0.0; else Vy = 0.0;
                    return;
                }
                PlayerX = nextX;
                PlayerY = nextY;
                remaining -= step;
            }
        }

        protected virtual void TryShoot()
        {
            if (ShootCooldown > 0 || AmmoCount <= 0) return;
            AmmoCount -= 1;
            ShootCooldown = ShootCooldownFrames;
            Play("shoot");
            var y = PlayerY + PlayerHeight * 0.45;
            var x = PlayerX + (Facing > 0 ? PlayerWidth : -8);
            Bullets.Add(new Bullet { X = x, Y = y, Vx = BulletSpeed * Facing, Ttl = 80, Powered = SuperTimer > 0 });
            AddVisualEvent("spark", x + (Facing > 0 ? 12 : -2), y, 10);
            if (!Grounded) Vx -= 0.35 * Facing;
        }

        protected void UpdateBullets()
        {
            var updated = new List<Bullet>(Bullets.Count);
            foreach (var bullet in Bullets)
            {
                bullet.X += bullet.Vx;
                bullet.Ttl -= 1;
                if (bullet.Ttl <= 0) continue;
                var tank = AirTankForRect(bullet.Rect);
                if (tank.HasValue)
                {
                    AirTanks.Remove(tank.Value);
                    Score = Math.Max(0, Score - 100);
                    AddTileEvent(tank.Value, "poof", "-AIR", 44);
                    DamageFromAirTank(tank.Value);
                    continue;
                }
                if (RectCollidesSolid(bullet.Rect))
                {
                    AddVisualEvent("spark", bullet.X, bullet.Y, 12);
                    continue;
                }
                updated.Add(bullet);
            }
            Bullets = updated;
        }

        /// <summary>
        /// The first tank a rectangle hits. Python iterates a set here; authored caves
        /// hold at most one tank (enforced by tests/test_sim_trace.py), so the sorted
        /// order used here cannot change the result.
        /// </summary>
        (int Col, int Row)? AirTankForRect(IntRect rect)
        {
            foreach (var tank in Sorted(AirTanks))
                if (rect.Collides(TileRect(tank).Inflate(-8, -6))) return tank;
            return null;
        }

        void DamageFromAirTank((int Col, int Row) tank)
        {
            var (tankX, tankY) = TileCenter(tank);
            var dx = tankX - (PlayerX + PlayerWidth / 2.0);
            var dy = tankY - (PlayerY + PlayerHeight / 2.0);
            if (Hypot(dx, dy) <= TileSize * 2.2) DamagePlayer();
        }

        /// <summary>Overflow-safe hypotenuse; equal to C hypot except possibly in the last bit.</summary>
        static double Hypot(double x, double y) => Math.Sqrt(x * x + y * y);

        protected void DamagePlayer()
        {
            if (InvulnTimer > 0) return;
            Health -= 1;
            InvulnTimer = InvulnFrames;
            ShakeTimer = ShakeFrames;
            Vy = -5.5 * GravityDir;
            Vx = -Facing * 2.0;
            AddVisualEvent("spark", PlayerX + PlayerWidth / 2.0, PlayerY + PlayerHeight / 2.0, 28, "OUCH");
            if (Health <= 0)
            {
                Health = 0;
                GameOver = true;
                Won = false;
                EndReason = "killed";
                Play("lose");
                return;
            }
            Play("damage");
        }

        // ---- feedback ----------------------------------------------------------------

        protected void AddVisualEvent(string kind, double x, double y, int ttl = 24, string text = "", int facing = 1) =>
            VisualEvents.Add(new VisualEvent { Kind = kind, X = x, Y = y, Ttl = ttl, MaxTtl = ttl, Text = text, Facing = facing });

        protected void AddTileEvent((int Col, int Row) tile, string kind, string text, int ttl, double yOffset = 0.0)
        {
            var (x, y) = TileCenter(tile);
            AddVisualEvent(kind, x, y + yOffset, ttl, text);
        }

        protected void UpdateVisualEvents()
        {
            if (VisualEvents.Count == 0) return;
            var updated = new List<VisualEvent>(VisualEvents.Count);
            foreach (var visual in VisualEvents)
            {
                visual.Ttl -= 1;
                if (visual.Ttl > 0) updated.Add(visual);
            }
            VisualEvents = updated;
        }

        public static List<(int Col, int Row)> Sorted(IEnumerable<(int Col, int Row)> tiles)
        {
            var list = new List<(int Col, int Row)>(tiles);
            list.Sort((a, b) => a.Col != b.Col ? a.Col.CompareTo(b.Col) : a.Row.CompareTo(b.Row));
            return list;
        }
    }
}

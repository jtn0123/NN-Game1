#nullable enable
using System.Collections.Generic;

namespace CrystalCaves.Sim
{
    /// <summary>
    /// Everything presentation needs after a command, in full double precision. Field
    /// names follow the Python protocol snapshot so the two can be compared directly.
    /// </summary>
    public sealed class SimSnapshot
    {
        public string Realm { get; set; } = "cave";
        public int ClearedCaves { get; set; }
        public int NearEntrance { get; set; } = -1;
        public int PortalLevel { get; set; } = -1;
        public int Episode { get; set; }
        public int Level { get; set; }
        public string LevelName { get; set; } = "";
        public IReadOnlyList<string> Levels { get; set; } = System.Array.Empty<string>();
        public IReadOnlyList<string> Layout { get; set; } = System.Array.Empty<string>();
        public int Cols { get; set; }
        public int Rows { get; set; }
        public SimPlayer Player { get; set; } = new SimPlayer();
        public List<SimEntity> Entities { get; } = new List<SimEntity>();
        public int Health { get; set; }
        public int Ammo { get; set; }
        public int Score { get; set; }
        public int Crystals { get; set; }
        public int InitialCrystals { get; set; }
        public int Steps { get; set; }
        public int FreezeTimer { get; set; }
        public int SuperTimer { get; set; }
        public bool ExitUnlocked { get; set; }
        public bool Done { get; set; }
        public bool Won { get; set; }
        public bool HumanOnly { get; set; }
        public string EndReason { get; set; } = "running";
        public string Mode { get; set; } = "human";
        public IReadOnlyList<string> Sounds { get; set; } = System.Array.Empty<string>();
        public List<SimEffect> Effects { get; } = new List<SimEffect>();
    }

    public sealed class SimPlayer
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Vx { get; set; }
        public double Vy { get; set; }
        public string Sprite { get; set; } = "mylo_idle";
        public int Facing { get; set; } = 1;
        public int GravityDir { get; set; } = 1;
        public int InvulnerabilityLeft { get; set; }
        public int InvulnerabilityFrames { get; set; }
        public bool Grounded { get; set; }
        public bool Invulnerable { get; set; }
        public bool Climbing { get; set; }
    }

    public enum EntityExtras { None, Creature, Lift }

    public sealed class SimEntity
    {
        public string Id { get; set; } = "";
        public string Sprite { get; set; } = "";
        public string Glow { get; set; } = "";
        public double X { get; set; }
        public double Y { get; set; }
        public double W { get; set; } = 32;
        public double H { get; set; } = 32;
        public bool Flip { get; set; }
        /// <summary>Which optional fields the reference protocol sends for this entity.</summary>
        public EntityExtras Extras { get; set; }
        public bool Asleep { get; set; }
        public bool Hit { get; set; }
        public int Frame { get; set; }
    }

    public sealed class SimEffect
    {
        /// <summary>Stable for the effect's lifetime; unique among live effects.</summary>
        public string Id { get; set; } = "";
        public string Kind { get; set; } = "";
        public string Text { get; set; } = "";
        public double X { get; set; }
        public double Y { get; set; }
        public int Ttl { get; set; }
        public int MaxTtl { get; set; }
        public int Facing { get; set; } = 1;
    }
}

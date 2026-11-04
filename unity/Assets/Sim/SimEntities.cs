#nullable enable
using System;
using System.Collections.Generic;

namespace CrystalCaves.Sim
{
    public sealed class CaveSpec
    {
        public string Name { get; }
        public IReadOnlyList<string> Layout { get; }

        public CaveSpec(string name, string[] layout)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            if (layout == null || layout.Length == 0) throw new ArgumentException("a cave needs rows", nameof(layout));
            Layout = Array.AsReadOnly((string[])layout.Clone());
        }

        /// <summary>The authored character, or '.' beyond a short row (Python ljust).</summary>
        public char At(int col, int row) => col < Layout[row].Length ? Layout[row][col] : '.';
    }

    public sealed class SecretSite
    {
        public string Cave { get; }
        public (int Col, int Row) Block { get; }
        public (int Col, int Row) Crystal { get; }

        public SecretSite(string cave, (int, int) block, (int, int) crystal)
        { Cave = cave; Block = block; Crystal = crystal; }
    }

    /// <summary>One frame of independent human buttons.</summary>
    public readonly struct HumanControls : IEquatable<HumanControls>
    {
        public int Move { get; }
        public bool Jump { get; }
        public bool Shoot { get; }
        public bool Interact { get; }

        public HumanControls(int move, bool jump, bool shoot, bool interact)
        {
            if (move < -1 || move > 1) throw new ArgumentOutOfRangeException(nameof(move), "move must be -1, 0 or 1");
            Move = move; Jump = jump; Shoot = shoot; Interact = interact;
        }

        public static readonly HumanControls Idle = new HumanControls(0, false, false, false);
        public bool Equals(HumanControls other) =>
            Move == other.Move && Jump == other.Jump && Shoot == other.Shoot && Interact == other.Interact;
        public override bool Equals(object? obj) => obj is HumanControls other && Equals(other);
        public override int GetHashCode() => (Move + 1) * 8 + (Jump ? 1 : 0) + (Shoot ? 2 : 0) + (Interact ? 4 : 0);
        public static bool operator ==(HumanControls left, HumanControls right) => left.Equals(right);
        public static bool operator !=(HumanControls left, HumanControls right) => !left.Equals(right);
    }

    internal sealed class Bullet
    {
        public double X, Y, Vx;
        public int Ttl;
        public bool Powered;
        public IntRect Rect => new IntRect(PyMath.Trunc(X), PyMath.Trunc(Y), 10, 4);
    }

    internal enum EnemyKind { Crawler, Flyer }

    internal static class Appearance
    {
        public const string Slug = "slug_enemy";
        public const string Rock = "walking_rock";
        public const string Dinosaur = "dinosaur_enemy";
        public const string Bat = "bat_enemy";
        public const string Eye = "eye_flyer";
    }

    /// <summary>Classic human-play creature (Python ClassicEnemy).</summary>
    internal sealed class Enemy
    {
        public double X, Y, Vx;
        public EnemyKind Kind;
        public bool Alive = true;
        public string Appearance = Sim.Appearance.Slug;
        public int Health = 1;
        public bool Asleep;
        public int WakeTimer;
        public bool Charging;
        public int HitUntil;
        /// <summary>Eye flyers gain a vertical patrol speed on their first move.</summary>
        public double? RoamVy;

        public const int Width = 24;
        public int Height => Appearance == Sim.Appearance.Dinosaur ? 64 : 24;
        public IntRect Rect => new IntRect(PyMath.Trunc(X), PyMath.Trunc(Y), Width, Height);
    }

    internal sealed class Elevator
    {
        public int Col, Top, Bottom;
        public double Pos;
        public int Direction = 1;
    }

    internal sealed class GreenThorn
    {
        public int Col, Row, Frame, Clock;
        public bool Triggered;
        public IntRect Rect => new IntRect(Col * 32 + 12, (Row + 1) * 32 - Frame * 8, 8, Frame * 8);
    }

    internal sealed class Stalactite
    {
        public int Col, Row, Y, Clock;
        public bool Falling, Alive = true;
        public IntRect Rect => new IntRect(Col * 32 + 10, Y + 2, 12, 30);
    }

    internal sealed class BatEgg
    {
        public int Identity;
        public double X, Y, Vy = 0.5;
        public int Ttl = 240;
        public IntRect Rect => new IntRect(PyMath.Trunc(X), PyMath.Trunc(Y), 12, 16);
    }

    /// <summary>Short-lived presentation feedback; never read back by the simulation.</summary>
    internal sealed class VisualEvent
    {
        public string Kind = "";
        public double X, Y;
        public int Ttl, MaxTtl;
        public string Text = "";
        public int Facing = 1;
    }
}

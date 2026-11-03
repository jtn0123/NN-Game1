#nullable enable
using System;

namespace CrystalCaves.Sim
{
    /// <summary>Integer rectangle with pygame 2 semantics, which the reference engine uses.</summary>
    public readonly struct IntRect : IEquatable<IntRect>
    {
        public IntRect(int x, int y, int w, int h) { X = x; Y = y; W = w; H = h; }

        public int X { get; }
        public int Y { get; }
        public int W { get; }
        public int H { get; }

        public int Left => X;
        public int Top => Y;
        public int Right => X + W;
        public int Bottom => Y + H;
        // pygame computes centers with C integer division (truncation toward zero).
        public int CenterX => X + W / 2;
        public int CenterY => Y + H / 2;

        public IntRect Offset(int dx, int dy) => new IntRect(X + dx, Y + dy, W, H);

        /// <summary>pygame Rect.inflate: grows (or shrinks) around the center.</summary>
        public IntRect Inflate(int dx, int dy) => new IntRect(X - dx / 2, Y - dy / 2, W + dx, H + dy);

        /// <summary>pygame Rect.colliderect: strict overlap; an empty rectangle never collides.</summary>
        public bool Collides(IntRect other)
        {
            if (W == 0 || H == 0 || other.W == 0 || other.H == 0) return false;
            return Math.Min(X, X + W) < Math.Max(other.X, other.X + other.W)
                && Math.Min(Y, Y + H) < Math.Max(other.Y, other.Y + other.H)
                && Math.Max(X, X + W) > Math.Min(other.X, other.X + other.W)
                && Math.Max(Y, Y + H) > Math.Min(other.Y, other.Y + other.H);
        }

        public bool Equals(IntRect other) => X == other.X && Y == other.Y && W == other.W && H == other.H;
        public override bool Equals(object? obj) => obj is IntRect other && Equals(other);
        public override int GetHashCode() => ((X * 397 ^ Y) * 397 ^ W) * 397 ^ H;
        public override string ToString() => $"<rect({X}, {Y}, {W}, {H})>";
        public static bool operator ==(IntRect left, IntRect right) => left.Equals(right);
        public static bool operator !=(IntRect left, IntRect right) => !left.Equals(right);
    }

    /// <summary>Python arithmetic that differs from C#'s defaults.</summary>
    public static class PyMath
    {
        /// <summary>Python <c>a // b</c> for integers: floors toward negative infinity.</summary>
        public static int FloorDiv(int a, int b)
        {
            if (b == 0) throw new DivideByZeroException();
            var quotient = a / b;
            return (a % b != 0 && (a < 0) != (b < 0)) ? quotient - 1 : quotient;
        }

        /// <summary>Python <c>int(x // 32)</c> for a float; exact because 32 is a power of two.</summary>
        public static int FloorDiv32(double value) => checked((int)Math.Floor(value / 32.0));

        /// <summary>Python <c>int(x)</c> for a float: truncation toward zero.</summary>
        public static int Trunc(double value)
        {
            if (double.IsNaN(value) || value >= 2147483648.0 || value <= -2147483649.0)
                throw new OverflowException($"cannot truncate {value} to int");
            return (int)value;
        }

        /// <summary>numpy.clip for finite values.</summary>
        public static double Clip(double value, double low, double high) => Math.Min(Math.Max(value, low), high);
    }
}

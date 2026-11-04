using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using CrystalCaves.Sim;

namespace CrystalCaves.SimTests
{
    /// <summary>
    /// The canonical snapshot encoding shared with <c>src/unity_bridge/sim_trace.py</c>.
    /// Values are <c>null</c>, <c>bool</c>, <c>long</c>, <c>double</c>, <c>string</c>,
    /// <c>List&lt;object?&gt;</c> or <c>SortedDictionary&lt;string, object?&gt;</c>.
    /// </summary>
    public static class Canonical
    {
        public static string Encode(object? value)
        {
            var builder = new StringBuilder();
            Write(builder, value);
            return builder.ToString();
        }

        public static string Digest(object? value)
        {
            var hash = SHA256.HashData(Encoding.UTF8.GetBytes(Encode(value)));
            return Convert.ToHexString(hash, 0, 8).ToLowerInvariant();
        }

        static void Write(StringBuilder builder, object? value)
        {
            switch (value)
            {
                case null: builder.Append("null"); break;
                case bool flag: builder.Append(flag ? "true" : "false"); break;
                case int integer: WriteNumber(builder, integer); break;
                case long integer: WriteNumber(builder, integer); break;
                case double number: WriteNumber(builder, number); break;
                case string text:
                    if (text.Any(ch => ch == '"' || ch == '\\' || ch < ' '))
                        throw new ArgumentException($"string cannot be encoded canonically: {text}");
                    builder.Append('"').Append(text).Append('"');
                    break;
                case IDictionary<string, object?> map:
                    builder.Append('{');
                    var first = true;
                    foreach (var key in map.Keys.OrderBy(key => key, StringComparer.Ordinal))
                    {
                        if (!first) builder.Append(',');
                        first = false;
                        builder.Append('"').Append(key).Append("\":");
                        Write(builder, map[key]);
                    }
                    builder.Append('}');
                    break;
                case System.Collections.IEnumerable items:
                    builder.Append('[');
                    var separator = false;
                    foreach (var item in items)
                    {
                        if (separator) builder.Append(',');
                        separator = true;
                        Write(builder, item);
                    }
                    builder.Append(']');
                    break;
                default: throw new ArgumentException($"unsupported canonical value {value.GetType().Name}");
            }
        }

        static void WriteNumber(StringBuilder builder, double number)
        {
            if (double.IsNaN(number) || double.IsInfinity(number)) throw new ArgumentException("non-finite number");
            if (number == Math.Floor(number) && Math.Abs(number) < 9007199254740992.0)
                builder.Append(((long)number).ToString(CultureInfo.InvariantCulture));
            else
                builder.Append('d').Append(((ulong)BitConverter.DoubleToInt64Bits(number)).ToString("x16", CultureInfo.InvariantCulture));
        }

        /// <summary>The comparable part of a snapshot, keyed exactly like the Python protocol.</summary>
        public static SortedDictionary<string, object?> From(SimSnapshot snapshot)
        {
            var player = snapshot.Player;
            var effects = snapshot.Effects.Select(effect => (object?)Map(
                ("kind", effect.Kind), ("x", effect.X), ("y", effect.Y), ("ttl", effect.Ttl),
                ("max_ttl", effect.MaxTtl), ("text", effect.Text), ("facing", effect.Facing))).ToList();
            effects.Sort((a, b) => string.CompareOrdinal(Encode(a), Encode(b)));
            return Map(
                ("protocol", 1), ("realm", snapshot.Realm), ("cleared_caves", snapshot.ClearedCaves),
                ("near_entrance", snapshot.NearEntrance), ("portal_level", snapshot.PortalLevel),
                ("episode", snapshot.Episode), ("level", snapshot.Level), ("level_name", snapshot.LevelName),
                ("levels", snapshot.Levels.Cast<object?>().ToList()), ("layout", snapshot.Layout.Cast<object?>().ToList()),
                ("cols", snapshot.Cols), ("rows", snapshot.Rows),
                ("player", Map(
                    ("x", player.X), ("y", player.Y), ("vx", player.Vx), ("vy", player.Vy),
                    ("sprite", player.Sprite), ("facing", player.Facing), ("grounded", player.Grounded),
                    ("gravity_dir", player.GravityDir), ("invulnerable", player.Invulnerable),
                    ("invulnerability_left", player.InvulnerabilityLeft),
                    ("invulnerability_frames", player.InvulnerabilityFrames), ("climbing", player.Climbing))),
                ("entities", snapshot.Entities.Select(EntityMap).Cast<object?>().ToList()),
                ("health", snapshot.Health), ("ammo", snapshot.Ammo), ("score", snapshot.Score),
                ("crystals", snapshot.Crystals), ("initial_crystals", snapshot.InitialCrystals),
                ("exit_unlocked", snapshot.ExitUnlocked), ("steps", snapshot.Steps),
                ("freeze_timer", snapshot.FreezeTimer), ("super_timer", snapshot.SuperTimer),
                ("done", snapshot.Done), ("won", snapshot.Won), ("end_reason", snapshot.EndReason),
                ("mode", snapshot.Mode), ("sounds", snapshot.Sounds.Cast<object?>().ToList()),
                ("human_only", snapshot.HumanOnly), ("effects", effects));
        }

        static SortedDictionary<string, object?> EntityMap(SimEntity entity)
        {
            var map = Map(
                ("id", entity.Id), ("sprite", entity.Sprite), ("x", entity.X), ("y", entity.Y),
                ("w", entity.W), ("h", entity.H), ("glow", entity.Glow), ("flip", entity.Flip));
            if (entity.Extras == EntityExtras.Creature)
            {
                map["asleep"] = entity.Asleep;
                map["hit"] = entity.Hit;
            }
            else if (entity.Extras == EntityExtras.Lift) map["frame"] = entity.Frame;
            return map;
        }

        static SortedDictionary<string, object?> Map(params (string Key, object? Value)[] pairs)
        {
            var map = new SortedDictionary<string, object?>(StringComparer.Ordinal);
            foreach (var (key, value) in pairs) map.Add(key, value);
            return map;
        }

        /// <summary>Parse a JSON checkpoint into the canonical value model.</summary>
        public static object? FromJson(JsonElement element)
        {
            switch (element.ValueKind)
            {
                case JsonValueKind.Null: return null;
                case JsonValueKind.True: return true;
                case JsonValueKind.False: return false;
                case JsonValueKind.String: return element.GetString();
                case JsonValueKind.Number:
                    return element.TryGetInt64(out var integer) ? integer : element.GetDouble();
                case JsonValueKind.Array: return element.EnumerateArray().Select(FromJson).ToList();
                case JsonValueKind.Object:
                    var map = new SortedDictionary<string, object?>(StringComparer.Ordinal);
                    foreach (var property in element.EnumerateObject()) map.Add(property.Name, FromJson(property.Value));
                    return map;
                default: throw new ArgumentException($"unexpected JSON {element.ValueKind}");
            }
        }

        /// <summary>Human-readable differences between two canonical values.</summary>
        public static List<string> Differences(object? expected, object? actual, string path = "")
        {
            var differences = new List<string>();
            if (expected is IDictionary<string, object?> left && actual is IDictionary<string, object?> right)
            {
                foreach (var key in left.Keys.Union(right.Keys).OrderBy(key => key, StringComparer.Ordinal))
                {
                    var child = path + "." + key;
                    var inLeft = left.TryGetValue(key, out var leftValue);
                    var inRight = right.TryGetValue(key, out var rightValue);
                    if (!inLeft) differences.Add($"{child}: unexpected {Encode(rightValue)}");
                    else if (!inRight) differences.Add($"{child}: missing (expected {Encode(leftValue)})");
                    else differences.AddRange(Differences(leftValue, rightValue, child));
                }
            }
            else if (expected is List<object?> a && actual is List<object?> b)
            {
                for (var index = 0; index < Math.Max(a.Count, b.Count); index++)
                {
                    var child = $"{path}[{index}]";
                    if (index >= a.Count) differences.Add($"{child}: unexpected {Encode(b[index])}");
                    else if (index >= b.Count) differences.Add($"{child}: missing (expected {Encode(a[index])})");
                    else differences.AddRange(Differences(a[index], b[index], child));
                }
            }
            else if (Encode(expected) != Encode(actual))
                differences.Add($"{path}: expected {Describe(expected)}, got {Describe(actual)}");
            return differences;
        }

        static string Describe(object? value) => value is double number
            ? number.ToString("R", CultureInfo.InvariantCulture)
            : Encode(value);
    }
}

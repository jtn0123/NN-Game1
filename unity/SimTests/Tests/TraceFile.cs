using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using CrystalCaves.Sim;

namespace CrystalCaves.SimTests
{
    /// <summary>One golden trace recorded by <c>python -m src.unity_bridge.sim_trace</c>.</summary>
    public sealed class TraceFile
    {
        public const int SupportedVersion = 1;

        public string Name { get; }
        public int Level { get; }
        public string InitialDigest { get; }
        public IReadOnlyList<TraceOperation> Operations { get; }
        public IReadOnlyList<string> Digests { get; }
        public IReadOnlyDictionary<int, object?> Checkpoints { get; }

        TraceFile(string name, int level, string initialDigest, List<TraceOperation> operations,
            List<string> digests, Dictionary<int, object?> checkpoints)
        {
            Name = name; Level = level; InitialDigest = initialDigest;
            Operations = operations; Digests = digests; Checkpoints = checkpoints;
        }

        public static string Directory([CallerFilePath] string source = "") =>
            Path.GetFullPath(Path.Combine(Path.GetDirectoryName(source)!, "..", "Traces"));

        public static IEnumerable<string> Names() => System.IO.Directory
            .GetFiles(Directory(), "*.json.gz")
            .Select(path => Path.GetFileName(path)[..^".json.gz".Length])
            .OrderBy(name => name, StringComparer.Ordinal);

        public static TraceFile Load(string name)
        {
            using var file = File.OpenRead(Path.Combine(Directory(), name + ".json.gz"));
            using var gzip = new GZipStream(file, CompressionMode.Decompress);
            using var document = JsonDocument.Parse(gzip);
            var root = document.RootElement;
            var version = root.GetProperty("version").GetInt32();
            if (version != SupportedVersion) throw new InvalidDataException($"{name}: trace version {version}");

            var operations = new List<TraceOperation>();
            foreach (var element in root.GetProperty("operations").EnumerateArray())
            {
                if (element.ValueKind == JsonValueKind.Array)
                {
                    var values = element.EnumerateArray().Select(value => value.GetInt32()).ToArray();
                    if (values.Length != 5 || values[0] < 1) throw new InvalidDataException($"{name}: bad run {element}");
                    var control = new HumanControls(values[1], values[2] != 0, values[3] != 0, values[4] != 0);
                    for (var repeat = 0; repeat < values[0]; repeat++) operations.Add(TraceOperation.Step(new[] { control }));
                }
                else operations.Add(TraceOperation.Parse(name, element));
            }

            var digests = root.GetProperty("digests").EnumerateArray().Select(value => value.GetString()!).ToList();
            if (digests.Count != operations.Count)
                throw new InvalidDataException($"{name}: {digests.Count} digests for {operations.Count} operations");
            var checkpoints = root.GetProperty("checkpoints").EnumerateObject()
                .ToDictionary(property => int.Parse(property.Name, System.Globalization.CultureInfo.InvariantCulture),
                    property => Canonical.FromJson(property.Value));
            return new TraceFile(root.GetProperty("name").GetString()!, root.GetProperty("level").GetInt32(),
                root.GetProperty("initial_digest").GetString()!, operations, digests, checkpoints);
        }
    }

    public enum TraceOperationKind { HumanStep, Reset, Mine }

    public sealed class TraceOperation
    {
        public TraceOperationKind Kind { get; }
        public IReadOnlyList<HumanControls> Controls { get; }
        public int Level { get; }
        public IReadOnlyList<int> Cleared { get; }

        TraceOperation(TraceOperationKind kind, IReadOnlyList<HumanControls> controls, int level, IReadOnlyList<int> cleared)
        { Kind = kind; Controls = controls; Level = level; Cleared = cleared; }

        public static TraceOperation Step(IReadOnlyList<HumanControls> controls) =>
            new TraceOperation(TraceOperationKind.HumanStep, controls, 0, Array.Empty<int>());

        public static TraceOperation Parse(string trace, JsonElement element)
        {
            var op = element.GetProperty("op").GetString();
            switch (op)
            {
                case "human_step":
                    return Step(element.GetProperty("controls").EnumerateArray().Select(control => new HumanControls(
                        control.GetProperty("move").GetInt32(), control.GetProperty("jump").GetBoolean(),
                        control.GetProperty("shoot").GetBoolean(), control.GetProperty("interact").GetBoolean())).ToArray());
                case "reset":
                    return new TraceOperation(TraceOperationKind.Reset, Array.Empty<HumanControls>(),
                        element.GetProperty("level").GetInt32(), Array.Empty<int>());
                case "mine":
                    return new TraceOperation(TraceOperationKind.Mine, Array.Empty<HumanControls>(), 0,
                        element.GetProperty("cleared").EnumerateArray().Select(value => value.GetInt32()).ToArray());
                default: throw new InvalidDataException($"{trace}: unsupported operation {element}");
            }
        }

        public SimSnapshot ApplyTo(CaveSim sim) => Kind switch
        {
            TraceOperationKind.HumanStep => sim.HumanStep(Controls),
            TraceOperationKind.Reset => sim.Reset(Level),
            _ => sim.OpenMine(Cleared),
        };

        public override string ToString() => Kind switch
        {
            TraceOperationKind.HumanStep => "step " + string.Join(" ", Controls.Select(c =>
                $"{c.Move:+0;-0;0}{(c.Jump ? "J" : "")}{(c.Shoot ? "S" : "")}{(c.Interact ? "I" : "")}")),
            TraceOperationKind.Reset => $"reset {Level}",
            _ => $"mine [{string.Join(",", Cleared)}]",
        };
    }
}

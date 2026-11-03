using System.Collections.Generic;
using System.Linq;
using System.Text;
using CrystalCaves.Sim;
using Xunit;

namespace CrystalCaves.SimTests
{
    /// <summary>
    /// The C# engine must reproduce every golden Python trace exactly: the same digest after
    /// every operation. On divergence the failure names the first bad operation and the
    /// field-level differences against the nearest checkpoint.
    /// </summary>
    public class ParityTests
    {
        public static IEnumerable<object[]> Traces() => TraceFile.Names().Select(name => new object[] { name });

        [Fact]
        public void Golden_traces_are_present()
        {
            Assert.True(TraceFile.Names().Count() >= 40, $"expected golden traces in {TraceFile.Directory()}");
        }

        [Theory]
        [MemberData(nameof(Traces))]
        public void Engine_matches_python_trace(string name)
        {
            var trace = TraceFile.Load(name);
            var sim = new CaveSim(trace.Level);
            var initial = Canonical.From(sim.Snapshot());
            if (Canonical.Digest(initial) != trace.InitialDigest)
                Assert.Fail(Report(trace, -1, null, initial));

            var recent = new Queue<string>();
            for (var index = 0; index < trace.Operations.Count; index++)
            {
                var operation = trace.Operations[index];
                var canonical = Canonical.From(operation.ApplyTo(sim));
                recent.Enqueue($"  #{index} {operation}");
                if (recent.Count > 6) recent.Dequeue();
                if (Canonical.Digest(canonical) != trace.Digests[index])
                    Assert.Fail(Report(trace, index, recent, canonical));
                if (trace.Checkpoints.TryGetValue(index, out var expected))
                {
                    var differences = Canonical.Differences(expected, canonical);
                    Assert.True(differences.Count == 0,
                        $"{name} checkpoint #{index} digest matched but fields differ:\n" + string.Join("\n", differences));
                }
            }
        }

        static string Report(TraceFile trace, int index, IEnumerable<string>? recent, object actual)
        {
            var text = new StringBuilder();
            text.AppendLine($"{trace.Name}: first divergence at operation #{index}");
            if (recent != null) text.AppendLine("recent operations:").AppendLine(string.Join("\n", recent));
            if (trace.Checkpoints.TryGetValue(index, out var expected))
            {
                text.AppendLine("differences:");
                foreach (var line in Canonical.Differences(expected, actual).Take(40)) text.AppendLine("  " + line);
            }
            else
            {
                var before = trace.Checkpoints.Keys.Where(key => key < index).DefaultIfEmpty(-1).Max();
                text.AppendLine($"no checkpoint here (nearest earlier: #{before}); actual state:");
                text.AppendLine(Canonical.Encode(actual));
            }
            return text.ToString();
        }
    }
}

using System.Collections.Generic;
using Xunit;

namespace CrystalCaves.SimTests
{
    /// <summary>Encoder cases mirrored in <c>tests/test_sim_trace.py</c>.</summary>
    public class CanonicalTests
    {
        [Theory]
        [InlineData(3.0, "3")]
        [InlineData(-0.0, "0")]
        [InlineData(0.5, "d3fe0000000000000")]
        [InlineData(140.0 / 60, "d4002aaaaaaaaaaab")]
        [InlineData(9007199254740992.0, "d4340000000000000")]
        public void Numbers_encode_exactly(double value, string expected) =>
            Assert.Equal(expected, Canonical.Encode(value));

        [Fact]
        public void Objects_sort_keys_and_lists_keep_order()
        {
            var value = new SortedDictionary<string, object?>
            {
                ["b"] = new List<object?> { 2L, null, true }, ["a"] = "x", ["B"] = false,
            };
            Assert.Equal("{\"B\":false,\"a\":\"x\",\"b\":[2,null,true]}", Canonical.Encode(value));
            Assert.Equal("c5115fed98b2d1a7", Canonical.Digest(value));
        }
    }
}

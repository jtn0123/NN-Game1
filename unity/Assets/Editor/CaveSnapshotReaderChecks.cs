using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace CrystalCaves.Pilot.Editor
{
    public static class CaveSnapshotReaderChecks
    {
        const string Malformed = "{";
        const string Room = "\"episode\":7,\"realm\":\"cave\",\"mode\":\"human\",\"cols\":1,\"rows\":1,\"layout\":[\"#\"],\"entities\":[],\"effects\":[]";
        const string Player = "\"player\":{\"x\":101,\"y\":673,\"sprite\":\"mylo_idle\"}";
        const string Valid = "{\"protocol\":1," + Room + "," + Player + "}";
        [Serializable] sealed class Report { public bool success; public bool legacyMalformedThrew, legacyMissingPlayerConstructed; public string[] checks; }

        // Explicitly reproduce the former CavePilot parsing line. This entry
        // must fail when the malformed packet reaches JsonUtility unchanged.
        public static void Before()
        {
            JsonUtility.FromJson<CaveSnapshot>(Malformed);
            throw new Exception("Legacy malformed snapshot was unexpectedly accepted");
        }

        public static void Run()
        {
            var checks = new List<string>();
            Action<bool, string> check = (ok, name) => {
                if (!ok) throw new Exception("Snapshot reader check failed: " + name);
                checks.Add(name);
            };
            var legacyThrew = false;
            try { JsonUtility.FromJson<CaveSnapshot>(Malformed); }
            catch (ArgumentException) { legacyThrew = true; }
            check(legacyThrew, "The previous unguarded parsing line throws on the malformed packet");
            var legacyMissingPlayer = JsonUtility.FromJson<CaveSnapshot>("{\"protocol\":1}").player != null;
            check(legacyMissingPlayer, "The previous constructor parser invents a missing nested player");
            check(!CaveSnapshotReader.TryRead(Malformed, out var snapshot) && snapshot == null,
                "Malformed JSON is rejected without an exception or partial snapshot");
            check(!CaveSnapshotReader.TryRead("null", out snapshot) && snapshot == null,
                "Null JSON cannot become a playable snapshot");
            check(!CaveSnapshotReader.TryRead("{\"protocol\":2,\"player\":{}}", out snapshot) && snapshot == null,
                "Unknown protocol versions are rejected");
            check(!CaveSnapshotReader.TryRead("{\"protocol\":1}", out snapshot) && snapshot == null,
                "A successful snapshot must include its player");
            check(!CaveSnapshotReader.TryRead("{\"protocol\":1,\"player\":null}", out snapshot) && snapshot == null,
                "An explicit null player cannot become a successful snapshot");
            check(!CaveSnapshotReader.TryRead("{\"protocol\":1," + Room + "}", out snapshot) && snapshot == null,
                "Complete room data cannot substitute for a missing player");
            check(!CaveSnapshotReader.TryRead("{\"protocol\":1," + Room + ",\"player\":{}}", out snapshot) && snapshot == null,
                "An empty invented player cannot become a successful snapshot");
            check(!CaveSnapshotReader.TryRead(Valid.Replace("\"cols\":1", "\"cols\":0"), out snapshot),
                "Room dimensions must be positive");
            check(!CaveSnapshotReader.TryRead(Valid.Replace("\"rows\":1", "\"rows\":2"), out snapshot),
                "The layout must contain every room row");
            check(!CaveSnapshotReader.TryRead(Valid.Replace("\"layout\":[\"#\"]", "\"layout\":[\"##\"]"), out snapshot),
                "Layout rows must match the room width");
            check(!CaveSnapshotReader.TryRead(Valid.Replace("\"entities\":[],", ""), out snapshot),
                "Successful snapshots must include their entity array");
            check(!CaveSnapshotReader.TryRead(Valid.Replace(",\"effects\":[]", ""), out snapshot),
                "Successful snapshots must include their effect array");
            check(CaveSnapshotReader.TryRead("{\"protocol\":1,\"error\":\"bad request\"}", out snapshot)
                && snapshot.error == "bad request" && snapshot.player == null,
                "A valid server error reaches the existing error branch without a player");
            check(CaveSnapshotReader.TryRead(Valid, out snapshot)
                && snapshot.episode == 7 && snapshot.player.x == 101 && snapshot.player.y == 673,
                "A valid snapshot is accepted after malformed, invalid and error packets");
            var args = Environment.GetCommandLineArgs(); var index = Array.IndexOf(args, "--snapshot-check-report");
            if (index >= 0 && index + 1 < args.Length)
                File.WriteAllText(args[index + 1], JsonUtility.ToJson(new Report {
                    success = true, legacyMalformedThrew = legacyThrew,
                    legacyMissingPlayerConstructed = legacyMissingPlayer, checks = checks.ToArray()
                }, true));
            Debug.Log("Passed " + checks.Count + " snapshot reader checks");
        }
    }
}

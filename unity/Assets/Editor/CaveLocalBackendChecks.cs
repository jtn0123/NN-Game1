using System;
using System.Collections.Generic;
using UnityEngine;

namespace CrystalCaves.Pilot.Editor
{
    // The standalone backend must hand CavePilot exactly what a valid bridge reply would.
    public static class CaveLocalBackendChecks
    {
        public static void Run()
        {
            var checks = new List<string>();
            Action<bool, string> check = (ok, name) => { if (!ok) throw new Exception("Local backend check failed: " + name); checks.Add(name); };
            using (var backend = new LocalCaveBackend())
            {
                CaveSnapshot Reply(CaveCommand command)
                {
                    check(backend.Send(command) && !backend.Busy && backend.Connected && backend.InProcess, "Commands are accepted synchronously");
                    check(backend.Receive(out var reply) && reply != null && !backend.Receive(out _), "Exactly one reply per command");
                    return reply;
                }
                bool Valid(CaveSnapshot snapshot) =>
                    CaveSnapshotReader.TryRead(JsonUtility.ToJson(snapshot), out var parsed) && string.IsNullOrEmpty(parsed.error);

                var first = Reply(new CaveCommand { op = "snapshot" });
                check(Valid(first) && first.realm == "cave" && first.levels.Length == 16 && first.mode == "human" && !first.ai_available, "Initial snapshot passes the bridge validator");
                var reset = Reply(new CaveCommand { op = "reset", level = 3 });
                check(Valid(reset) && reset.level == 3 && reset.steps == 0 && Array.IndexOf(reset.sounds, "enter") >= 0, "Reset opens the requested cave with its entry cue");
                var right = new CaveHumanControl { move = 1 };
                var step = Reply(new CaveCommand { op = "human_step", controls = new[] { right, right, right } });
                check(Valid(step) && step.steps == 3 && step.player.x > reset.player.x, "Human steps advance the authoritative cave");
                var mine = Reply(new CaveCommand { op = "mine", cleared = new[] { 0, 2 } });
                check(Valid(mine) && mine.realm == "mine" && mine.cleared_caves == 2, "The main mine opens with cleared caves");
                check(!string.IsNullOrEmpty(Reply(new CaveCommand { op = "reset", level = 16 }).error), "Out-of-range caves are rejected");
                check(Reply(new CaveCommand { op = "mode", mode = "ai" }).error == LocalCaveBackend.AiUnavailable, "AI mode explains that it needs the Python checkpoint");
                check(!string.IsNullOrEmpty(Reply(new CaveCommand { op = "human_step", controls = new CaveHumanControl[9] }).error), "Oversized batches are rejected");
                check(!string.IsNullOrEmpty(Reply(new CaveCommand { op = "step", actions = new[] { 2 } }).error), "Numbered AI actions are rejected in standalone play");
                check(Valid(Reply(new CaveCommand { op = "snapshot" })), "The backend stays usable after rejected commands");
            }
            Debug.Log("Passed " + checks.Count + " local backend checks");
        }
    }
}

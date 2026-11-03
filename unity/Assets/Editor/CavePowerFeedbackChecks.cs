using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace CrystalCaves.Pilot.Editor
{
    public static class CavePowerFeedbackChecks
    {
        [Serializable] sealed class Report { public bool success; public string[] checks; }
        public static void Run()
        {
            var checks = new List<string>();
            Action<bool, string> check = (ok, name) => {
                if (!ok) throw new Exception("Power feedback check failed: " + name);
                checks.Add(name);
            };
            var frames = new[] { 0, 1, 59, 60, 61, 420 };
            var seconds = new[] { 0, 1, 1, 1, 2, 7 };
            for (var i = 0; i < frames.Length; i++)
                check(CavePowerFeedback.SecondsLeft(frames[i]) == seconds[i], frames[i] + " simulation ticks show " + seconds[i] + " seconds");
            check(CavePowerFeedback.PowerText(420) == "POWER 07S" && CavePowerFeedback.FreezeText(300) == "FREEZE 05S",
                "Pickup durations have clear named countdowns");
            check(CavePowerFeedback.PowerText(0) == "" && CavePowerFeedback.FreezeText(-1) == "",
                "Expired and invalid timers have no active label");
            check(CavePowerFeedback.PowerText(int.MaxValue) == "POWER 99+S",
                "Unusually large timer payloads remain compact without overflow");
            var old = JsonUtility.FromJson<CaveSnapshot>("{\"realm\":\"cave\",\"freeze_timer\":61}");
            check(old.super_timer == 0 && CavePowerFeedback.Rows(old) == 1 && CavePowerFeedback.FreezeText(old.freeze_timer) == "FREEZE 02S",
                "Older snapshots without super_timer do not invent an active power");
            check(CavePowerFeedback.Rows(new CaveSnapshot { realm = "mine", super_timer = 420, freeze_timer = 300 }) == 0,
                "Power status never covers the main-mine door hint");
            check(CavePowerFeedback.PickupLabel("sparkle", "P") == "POWER SHOTS"
                && CavePowerFeedback.PickupLabel("sparkle", "Z") == "FREEZE"
                && CavePowerFeedback.PickupLabel("score", "P") == "P"
                && CavePowerFeedback.PickupLabel("sparkle", "+100") == "+100", "Only power pickup letters gain explanatory labels");
            check(CaveSecretCacheFeedback.Marker(new CaveEntity { id = "secret_cache_6_8", sprite = "secret_cache_armed" }) == "?",
                "Item-symbol accessibility mode can identify an armed secret cache");
            check(CaveSecretCacheFeedback.Marker(new CaveEntity { id = "secret_cache_6_8", sprite = "secret_cache_empty" }) == null
                && CaveSecretCacheFeedback.Marker(new CaveEntity { id = "crystal_0", sprite = "secret_cache_armed" }) == null
                && CaveSecretCacheFeedback.Marker(null) == null, "Empty caches and unrelated objects do not gain a secret marker");
            var state = new CaveSnapshot { realm = "cave", super_timer = 420, freeze_timer = 300 };
            foreach (var scale in new[] { 1, 2 })
            foreach (var margin in new[] { 0, 24 })
            foreach (var bottomHud in new[] { true, false })
            {
                var hudHeight = scale == 2 ? 48 : 32;
                var worldTop = bottomHud ? 0 : hudHeight;
                var worldBottom = bottomHud ? 384 - hudHeight : 384;
                var bounds = CavePowerFeedback.Bounds(state, 640, worldTop, worldBottom, scale, margin);
                var hint = new Rect(8, worldTop + 4, 624, 17);
                var exitMessage = new Rect(255, bottomHud ? worldBottom - 16 : hudHeight + 16, 144, 8);
                check(bounds.xMin >= 8 + margin && bounds.xMax <= 232 && bounds.yMin >= worldTop
                    && bounds.yMax <= worldBottom - 4 && !bounds.Overlaps(hint) && !bounds.Overlaps(exitMessage),
                    "640x384 " + (bottomHud ? "bottom" : "top") + " HUD scale " + scale + " margin " + margin + " fits without hint/exit overlap");
            }
            state.super_timer = 0; state.freeze_timer = 0;
            check(CavePowerFeedback.Bounds(state, 640, 0, 352, 1, 0) == Rect.zero,
                "Inactive state contributes no status plate");
            var args = Environment.GetCommandLineArgs(); var index = Array.IndexOf(args, "--power-feedback-check-report");
            if (index >= 0 && index + 1 < args.Length)
                File.WriteAllText(args[index + 1], JsonUtility.ToJson(new Report { success = true, checks = checks.ToArray() }, true));
            Debug.Log("Passed " + checks.Count + " power feedback behavior checks");
        }
    }
}

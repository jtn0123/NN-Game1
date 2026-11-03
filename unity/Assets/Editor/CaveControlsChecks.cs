using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace CrystalCaves.Pilot.Editor
{
    public static class CaveControlsChecks
    {
        [Serializable] sealed class Report { public bool success; public string[] checks; }
        public static void Run()
        {
            var checks = new List<string>();
            Action<bool, string> check = (ok, name) => {
                if (!ok) throw new Exception("Controls check failed: " + name);
                checks.Add(name);
            };
            var data = new CaveSettingsData();
            check(CaveControls.Label(data, CaveControl.Jump) == "SPACE / W / UP"
                && CaveControls.Label(data, CaveControl.Shoot) == "J / X", "Default guidance includes all existing keyboard alternatives");
            check(!CaveControls.TryBind(data, CaveControl.Shoot, KeyCode.LeftArrow, out var message)
                && message.Contains("MOVE LEFT") && data.keyShoot == 0, "A conflicting alias is rejected without changing controls");
            check(!CaveControls.TryBind(data, CaveControl.Jump, KeyCode.F12, out message)
                && data.keyJump == 0, "Global screenshot keys cannot become gameplay bindings");
            check(!CaveControls.TryBind(data, CaveControl.Interact, KeyCode.Return, out message)
                && !CaveControls.IsBindable(KeyCode.JoystickButton0) && !CaveControls.IsBindable(KeyCode.Mouse0), "Menu confirmation and non-keyboard controls stay reserved");
            check(CaveControls.TryBind(data, CaveControl.Jump, KeyCode.LeftControl, out message)
                && CaveControls.TryBind(data, CaveControl.Shoot, KeyCode.LeftAlt, out message), "Modifier keys support alternative jump and fire layouts");
            var roundTrip = CaveSettings.Parse(JsonUtility.ToJson(data));
            check(roundTrip.keyJump == (int)KeyCode.LeftControl && roundTrip.keyShoot == (int)KeyCode.LeftAlt,
                "Custom keyboard controls survive saved-settings round trips");
            roundTrip.ApplyPreset(2);
            check(roundTrip.keyJump == (int)KeyCode.LeftControl && roundTrip.keyShoot == (int)KeyCode.LeftAlt,
                "Graphics presets preserve the player's controls");
            check(CaveControls.TryBind(data, CaveControl.Pause, KeyCode.K, out message)
                && CaveControls.Label(data, CaveControl.Pause) == "K / ESC", "Escape remains an explicit pause fallback after rebinding");
            data.keyShoot = (int)KeyCode.LeftArrow; data.Sanitize();
            check(data.keyJump == 0 && data.keyShoot == 0 && data.keyPause == 0, "A corrupted conflicting map restores usable defaults");
            data.keyJump = -123; data.keyInteract = (int)KeyCode.F2; data.Sanitize();
            check(data.keyJump == 0 && data.keyInteract == 0, "Invalid and reserved saved keys fall back safely");
            CaveControls.TryBind(data, CaveControl.Jump, KeyCode.K, out message); CaveControls.Reset(data);
            check(CaveControls.Label(data, CaveControl.Jump) == "SPACE / W / UP", "Restore defaults removes custom overrides");
            var command = new CaveCommand { op = "human_step", controls = new[] {
                new CaveHumanControl { move = 1, jump = true, shoot = true, interact = false }
            } };
            var commandJson = JsonUtility.ToJson(command);
            var wire = JsonUtility.FromJson<CaveCommand>(commandJson);
            check(wire.op == "human_step" && wire.controls.Length == 1 && wire.controls[0].move == 1
                && wire.controls[0].jump && wire.controls[0].shoot && !wire.controls[0].interact
                && commandJson.Contains("\"interact\":false"), "Independent jump/fire and neutral interaction survive the JSON wire format");
            var args = Environment.GetCommandLineArgs(); var index = Array.IndexOf(args, "--controls-check-report");
            if (index >= 0 && index + 1 < args.Length)
                File.WriteAllText(args[index + 1], JsonUtility.ToJson(new Report { success = true, checks = checks.ToArray() }, true));
            Debug.Log("Passed " + checks.Count + " controls behavior checks");
        }
    }
}

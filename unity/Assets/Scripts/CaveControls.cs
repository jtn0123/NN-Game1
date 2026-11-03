using System;
using UnityEngine;

namespace CrystalCaves.Pilot
{
    public enum CaveControl { MoveLeft, MoveRight, Jump, Shoot, Interact, Pause }

    // Only the keyboard changes here. Joystick axes/buttons keep their existing mapping.
    public static class CaveControls
    {
        static readonly KeyCode[][] defaults = {
            new[] { KeyCode.A, KeyCode.LeftArrow },
            new[] { KeyCode.D, KeyCode.RightArrow },
            new[] { KeyCode.Space, KeyCode.W, KeyCode.UpArrow },
            new[] { KeyCode.J, KeyCode.X },
            new[] { KeyCode.E, KeyCode.DownArrow },
            new[] { KeyCode.Escape, KeyCode.P }
        };
        public static readonly CaveControl[] Actions = (CaveControl[])Enum.GetValues(typeof(CaveControl));
        static readonly KeyCode[] keyboard = (KeyCode[])Enum.GetValues(typeof(KeyCode));

        public static bool Held(CaveControl action) => Read(action, false);
        public static bool Pressed(CaveControl action) => Read(action, true);
        static bool Read(CaveControl action, bool pressed)
        {
            var key = Configured(CaveSettings.Data, action);
            if (key != KeyCode.None)
                return ReadKey(key, pressed) || action == CaveControl.Pause && ReadKey(KeyCode.Escape, pressed);
            foreach (var alternative in defaults[(int)action]) if (ReadKey(alternative, pressed)) return true;
            return false;
        }
        static bool ReadKey(KeyCode key, bool pressed) => pressed ? Input.GetKeyDown(key) : Input.GetKey(key);
        public static string Name(CaveControl action)
        {
            switch (action)
            {
                case CaveControl.MoveLeft: return "MOVE LEFT";
                case CaveControl.MoveRight: return "MOVE RIGHT";
                case CaveControl.Jump: return "JUMP / CLIMB UP";
                case CaveControl.Shoot: return "FIRE RAYGUN";
                case CaveControl.Interact: return "USE / CLIMB DOWN";
                default: return "PAUSE";
            }
        }
        public static string KeyName(KeyCode key)
        {
            switch (key)
            {
                case KeyCode.LeftArrow: return "LEFT";
                case KeyCode.RightArrow: return "RIGHT";
                case KeyCode.UpArrow: return "UP";
                case KeyCode.DownArrow: return "DOWN";
                case KeyCode.LeftControl: return "L CTRL";
                case KeyCode.RightControl: return "R CTRL";
                case KeyCode.LeftAlt: return "L ALT";
                case KeyCode.RightAlt: return "R ALT";
                case KeyCode.LeftShift: return "L SHIFT";
                case KeyCode.RightShift: return "R SHIFT";
                case KeyCode.LeftCommand: return "L CMD";
                case KeyCode.RightCommand: return "R CMD";
                case KeyCode.Escape: return "ESC";
            }
            var name = key.ToString();
            if (name.StartsWith("Alpha", StringComparison.Ordinal)) return name.Substring(5);
            if (name.StartsWith("Keypad", StringComparison.Ordinal)) return "KP " + name.Substring(6).ToUpperInvariant();
            return name.ToUpperInvariant();
        }
        public static string Label(CaveControl action) => Label(CaveSettings.Data, action);
        public static string Label(CaveSettingsData data, CaveControl action)
        {
            var key = Configured(data, action);
            if (key != KeyCode.None) return KeyName(key) + (action == CaveControl.Pause ? " / ESC" : "");
            var names = new string[defaults[(int)action].Length];
            for (var i = 0; i < names.Length; i++) names[i] = KeyName(defaults[(int)action][i]);
            return string.Join(" / ", names);
        }
        public static string Compact(CaveControl action)
        {
            var key = Configured(CaveSettings.Data, action);
            if (key != KeyCode.None) return KeyName(key);
            return action == CaveControl.Shoot ? "J/X" : KeyName(defaults[(int)action][0]);
        }
        public static string Hint => Compact(CaveControl.MoveLeft) + "/" + Compact(CaveControl.MoveRight) + " MOVE  "
            + Compact(CaveControl.Jump) + " JUMP  " + Compact(CaveControl.Shoot) + " FIRE  "
            + Compact(CaveControl.Interact) + " USE  " + Compact(CaveControl.Pause) + " PAUSE";

        public static bool IsBindable(KeyCode key) => Enum.IsDefined(typeof(KeyCode), key) && (int)key > 0 && (int)key < (int)KeyCode.Mouse0
            && key != KeyCode.Escape && key != KeyCode.Return && key != KeyCode.KeypadEnter
            && key != KeyCode.M && key != KeyCode.F2 && key != KeyCode.F11 && key != KeyCode.F12;

        public static bool TryBind(CaveSettingsData data, CaveControl action, KeyCode key, out string message)
        {
            if (!IsBindable(key))
            { message = "That key is reserved. Try another key. ESC cancels."; return false; }
            foreach (var other in Actions)
            {
                if (other == action || !Uses(data, other, key)) continue;
                message = KeyName(key) + " is already used for " + Name(other) + ". Try another key.";
                return false;
            }
            Set(data, action, key); message = Name(action) + " saved: " + KeyName(key) + "."; return true;
        }
        public static bool TryBind(CaveControl action, KeyCode key, out string message)
        {
            if (!TryBind(CaveSettings.Data.Copy(), action, key, out message)) return false;
            CaveSettings.Change(d => Set(d, action, key), false); return true;
        }
        public static bool Capture(out KeyCode key)
        {
            foreach (var candidate in keyboard)
                if ((int)candidate > 0 && (int)candidate < (int)KeyCode.Mouse0 && Input.GetKeyDown(candidate))
                { key = candidate; return true; }
            key = KeyCode.None; return false;
        }
        public static void Reset(CaveSettingsData data)
        { foreach (var action in Actions) Set(data, action, KeyCode.None); }
        public static void Sanitize(CaveSettingsData data)
        {
            foreach (var action in Actions)
                if (Configured(data, action) != KeyCode.None && !IsBindable(Configured(data, action))) Set(data, action, KeyCode.None);
            // Fall back together if a saved map contains conflicts; every action stays usable.
            foreach (var action in Actions)
            {
                var key = Configured(data, action);
                if (key == KeyCode.None) continue;
                foreach (var other in Actions)
                    if (other != action && Uses(data, other, key)) { Reset(data); return; }
            }
        }
        static bool Uses(CaveSettingsData data, CaveControl action, KeyCode key)
        {
            var configured = Configured(data, action);
            return configured != KeyCode.None ? configured == key || action == CaveControl.Pause && key == KeyCode.Escape
                : Array.IndexOf(defaults[(int)action], key) >= 0;
        }
        static KeyCode Configured(CaveSettingsData data, CaveControl action)
        {
            switch (action)
            {
                case CaveControl.MoveLeft: return (KeyCode)data.keyMoveLeft;
                case CaveControl.MoveRight: return (KeyCode)data.keyMoveRight;
                case CaveControl.Jump: return (KeyCode)data.keyJump;
                case CaveControl.Shoot: return (KeyCode)data.keyShoot;
                case CaveControl.Interact: return (KeyCode)data.keyInteract;
                default: return (KeyCode)data.keyPause;
            }
        }
        static void Set(CaveSettingsData data, CaveControl action, KeyCode key)
        {
            switch (action)
            {
                case CaveControl.MoveLeft: data.keyMoveLeft = (int)key; break;
                case CaveControl.MoveRight: data.keyMoveRight = (int)key; break;
                case CaveControl.Jump: data.keyJump = (int)key; break;
                case CaveControl.Shoot: data.keyShoot = (int)key; break;
                case CaveControl.Interact: data.keyInteract = (int)key; break;
                case CaveControl.Pause: data.keyPause = (int)key; break;
            }
        }
    }
}

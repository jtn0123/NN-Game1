using System;
using System.Globalization;
using UnityEngine;

namespace CrystalCaves.Pilot
{
    public static class CavePowerFeedback
    {
        // Timers come from the authoritative 60 Hz simulation, including while paused.
        public static int SecondsLeft(int frames) => frames <= 0 ? 0 : (frames - 1) / 60 + 1;
        public static string PowerText(int frames) => Countdown("POWER", frames);
        public static string FreezeText(int frames) => Countdown("FREEZE", frames);
        static string Countdown(string name, int frames)
        {
            if (frames <= 0) return "";
            var seconds = SecondsLeft(frames);
            return name + " " + (seconds > 99 ? "99+" : seconds.ToString("D2", CultureInfo.InvariantCulture)) + "S";
        }
        public static string PickupLabel(string kind, string text)
        {
            if (kind == "sparkle")
            {
                if (text == "P") return "POWER SHOTS";
                if (text == "Z") return "FREEZE";
            }
            return text;
        }
        public static int Rows(CaveSnapshot state) => state == null || state.realm == "mine" ? 0
            : (state.super_timer > 0 ? 1 : 0) + (state.freeze_timer > 0 ? 1 : 0);
        public static int TextScale(int hudScale) => hudScale > 1 ? 2 : 1;
        public static Rect Bounds(CaveSnapshot state, float logicalWidth, float worldTop, float worldBottom, int hudScale, int safeMargin)
        {
            var rows = Rows(state);
            if (rows == 0) return Rect.zero;
            var scale = TextScale(hudScale);
            var length = Math.Max(PowerText(state.super_timer).Length, FreezeText(state.freeze_timer).Length);
            var left = 8 + Math.Min(24, Math.Max(0, safeMargin));
            // Stay left of the existing exit-open message, even in a 640 px viewport.
            var width = Math.Min(length * 6 * scale + 12, Math.Max(0, Math.Min(logicalWidth - 8, 232) - left));
            var height = rows * (8 * scale + 4) + 4;
            return new Rect(left, Math.Max(worldTop + 32, worldBottom - height - 4), width, height);
        }
    }
}

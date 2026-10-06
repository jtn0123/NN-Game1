namespace CrystalCaves.Pilot
{
    public static class CaveVisualSettings
    {
        public static bool Motion => !CaveSettings.Data.reducedMotion;
        public static bool HudBottom => CaveSettings.Data.hudBottom;
        public static void ToggleHud() => CaveSettings.Change(d => d.hudBottom = !d.hudBottom, false);
        public static void ToggleMotion() => CaveSettings.Change(d => d.reducedMotion = !d.reducedMotion, false);
    }
}

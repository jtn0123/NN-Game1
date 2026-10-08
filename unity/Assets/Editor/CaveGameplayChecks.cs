namespace CrystalCaves.Pilot.Editor
{
    // Run the game-facing protocol, input, HUD and settings contracts together.
    public static class CaveGameplayChecks
    {
        public static void Run()
        {
            CaveControlsChecks.Run();
            CavePowerFeedbackChecks.Run();
            CaveSettingsChecks.Run();
            CaveSnapshotReaderChecks.Run();
        }
    }
}

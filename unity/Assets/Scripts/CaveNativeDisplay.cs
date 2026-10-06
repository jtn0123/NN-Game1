using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace CrystalCaves.Pilot
{
    public static class CaveNativeDisplay
    {
        [DllImport("CaveDisplay")] static extern int CaveSetDisplaySync(int enabled);
        [DllImport("CaveDisplay")] static extern IntPtr CaveWindowState();
        [DllImport("CaveDisplay")] static extern int CaveWindowActive();
        [DllImport("CaveDisplay")] static extern int CaveWindowMatchesSize(int width, int height);
        public static bool WindowActive => !IsMacPlayer || CaveWindowActive() != 0;
        public static bool WindowMatchesSize(int width, int height) => !IsMacPlayer || CaveWindowMatchesSize(width, height) != 0;
        public static string WindowState => IsMacPlayer ? Marshal.PtrToStringAnsi(CaveWindowState()) : "Other platform";
        public static bool IsMacPlayer => Application.platform == RuntimePlatform.OSXPlayer;
        public static int Status { get; private set; }
        static bool failed;
        public static void Apply(bool enabled)
        {
            if (!IsMacPlayer || failed) return;
            try { Status = CaveSetDisplaySync(enabled ? 1 : 0); }
            catch (Exception error) when (error is DllNotFoundException || error is EntryPointNotFoundException || error is BadImageFormatException)
            { failed = true; Debug.LogError("Native display synchronization is unavailable: " + error.Message); }
        }
    }
}

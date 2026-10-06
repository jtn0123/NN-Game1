using System;
using System.Collections;
using System.IO;
using UnityEngine;

namespace CrystalCaves.Pilot
{
    public static class CaveVisualClock
    {
        public static bool Frozen;
        public static float FrozenAt;
        public static float Now => Frozen ? FrozenAt : Time.unscaledTime;
    }
    public sealed partial class CavePilot
    {
        CaveScreen screenshotReturn;
        bool screenshotHud;
        public bool ScreenshotMode { get; private set; }
        public bool CapturingScreenshot { get; private set; }
        public bool HideHud => ScreenshotMode && !screenshotHud;
        public float ScreenshotToastUntil { get; private set; }
        public string ScreenshotMessage { get; private set; }
        public string ScreenshotDirectory => CaveSettings.Temporary
            ? Path.Combine(Application.temporaryCachePath, "PresentationReview", "Screenshots")
            : Path.Combine(Application.persistentDataPath, "Screenshots");
        public void StartScreenshotMode()
        {
            if (state == null || presentation.Pending || CapturingScreenshot) return;
            screenshotReturn = Screen; screenshotHud = !CaveSettings.Data.screenshotHideHud;
            CaveViewport.HideHudStrip = !screenshotHud;
            ScreenshotMode = true; Screen = CaveScreen.Play; LabVisible = false; accumulator = 0;
            queuedJump = queuedShoot = queuedInteract = false;
            CaveVisualClock.FrozenAt = Time.unscaledTime; CaveVisualClock.Frozen = true;
        }
        public void ExitScreenshotMode()
        {
            if (CapturingScreenshot) return;
            CaveViewport.HideHudStrip = false;
            ScreenshotMode = false; Screen = screenshotReturn; CaveVisualClock.Frozen = false; accumulator = 0;
        }
        void HandleScreenshotKeys()
        {
            if (Input.GetKeyDown(KeyCode.Escape) || Input.GetKeyDown(KeyCode.JoystickButton1)) ExitScreenshotMode();
            if (Input.GetKeyDown(KeyCode.H) || Input.GetKeyDown(KeyCode.JoystickButton2)) { screenshotHud = !screenshotHud; CaveViewport.HideHudStrip = !screenshotHud; }
            if (Input.GetKeyDown(KeyCode.F12) || Input.GetKeyDown(KeyCode.JoystickButton0)) SaveScreenshot();
        }
        public void SaveScreenshot()
        { if (!CapturingScreenshot) StartCoroutine(CaptureGameScreenshot()); }
        IEnumerator CaptureGameScreenshot()
        {
            CapturingScreenshot = true;
            yield return new WaitForEndOfFrame();
            Texture2D texture = null;
            try
            {
                Directory.CreateDirectory(ScreenshotDirectory);
                var path = Path.Combine(ScreenshotDirectory, "Crystal-Caves-" + DateTime.Now.ToString("yyyyMMdd-HHmmss-fff") + "-" + Guid.NewGuid().ToString("N").Substring(0,6) + ".png");
                texture = ScreenCapture.CaptureScreenshotAsTexture();
                File.WriteAllBytes(path, texture.EncodeToPNG());
                ScreenshotMessage = "Screenshot saved\n" + path;
            }
            catch (Exception error) { ScreenshotMessage = "Screenshot could not be saved: " + error.Message; Debug.LogException(error); }
            finally { if (texture != null) Destroy(texture); CapturingScreenshot = false; ScreenshotToastUntil = Time.unscaledTime + 7; }
        }
    }
}

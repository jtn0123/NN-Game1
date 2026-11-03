using System;
using UnityEngine;

namespace CrystalCaves.Pilot
{
    [Serializable]
    public sealed class CaveSettingsData
    {
        public int version = 1, width = 1600, height = 1000;
        public bool fullscreen, vSync, showFps;
        public int frameCap = 60, pixelScale, aspect;
        public int shake = 100, effectDensity = 2, lighting, brightness = 100, palette;
        public bool particles = true, gemGlints = true, damageFlashes = true, environment;
        public bool contactShadows = true;
        public int shadowStrength = 45, scanlines, phosphor;
        public bool hudBottom = true, highContrast, itemMarkers, reducedMotion, screenshotHideHud;
        public int hudScale = 1, safeMargin;
        public int masterVolume = 70, effectsVolume = 100;
        public bool muted;
        // Zero keeps the familiar keyboard alternatives for that action.
        public int keyMoveLeft, keyMoveRight, keyJump, keyShoot, keyInteract, keyPause;
        public bool controlsHintSeen;
        public string preset = "Current Retro";

        public CaveSettingsData Copy() => JsonUtility.FromJson<CaveSettingsData>(JsonUtility.ToJson(this));
        public void Sanitize()
        {
            version = 1;
            width = Mathf.Clamp(width, 640, 7680); height = Mathf.Clamp(height, 384, 4320);
            frameCap = frameCap == 0 ? 0 : Mathf.Clamp(frameCap, 30, 360);
            pixelScale = Mathf.Clamp(pixelScale, 0, 6); aspect = Mathf.Clamp(aspect, 0, 2);
            shake = Mathf.Clamp(shake, 0, 100); effectDensity = Mathf.Clamp(effectDensity, 0, 2);
            lighting = Mathf.Clamp(lighting, 0, 50); brightness = Mathf.Clamp(brightness, 70, 130);
            palette = Mathf.Clamp(palette, 0, 2); shadowStrength = Mathf.Clamp(shadowStrength, 0, 80);
            scanlines = Mathf.Clamp(scanlines, 0, 60); phosphor = Mathf.Clamp(phosphor, 0, 40);
            hudScale = Mathf.Clamp(hudScale, 1, 2); safeMargin = Mathf.Clamp(safeMargin, 0, 24);
            masterVolume = Mathf.Clamp(masterVolume, 0, 100); effectsVolume = Mathf.Clamp(effectsVolume, 0, 100);
            CaveControls.Sanitize(this);
            if (preset != "Current Retro" && preset != "Enhanced Retro" && preset != "Low Power") preset = "Custom";
        }

        // Display, sound, HUD and accessibility preferences survive quality presets.
        public void ApplyPreset(int index)
        {
            shake = 100; particles = gemGlints = damageFlashes = contactShadows = true;
            effectDensity = 2; shadowStrength = 45; environment = false;
            lighting = scanlines = phosphor = palette = 0; brightness = 100;
            vSync = false; frameCap = 60;
            preset = index == 1 ? "Enhanced Retro" : index == 2 ? "Low Power" : "Current Retro";
            if (index == 1) { lighting = 25; environment = true; }
            if (index == 2)
            { vSync = false; frameCap = 30; shake = 0; effectDensity = 0; contactShadows = false; gemGlints = false; }
        }
    }

    public static class CaveSettings
    {
        public const string Key = "cave-settings-v1";
        public static CaveSettingsData Data { get; private set; } = new CaveSettingsData();
        public static bool Temporary { get; private set; }
        public static event Action Changed;
        public static void Load(bool temporary, string json = null)
        {
            Temporary = temporary;
            if (json != null) Data = Parse(json);
            else if (!temporary && PlayerPrefs.HasKey(Key)) Data = Parse(PlayerPrefs.GetString(Key));
            else
            {
                Data = new CaveSettingsData();
                if (!temporary)
                {
                    Data = FromLegacy(PlayerPrefs.GetInt("cave-hud-position",0), PlayerPrefs.GetInt("cave-reduced-motion",0), PlayerPrefs.GetFloat("cave-volume",.7f), PlayerPrefs.GetInt("cave-muted",0));
                }
            }
            Data.Sanitize();
            ApplyFramePacing();
        }
        public static CaveSettingsData FromLegacy(int hudPosition, int reducedMotion, float volume, int muted)
        {
            var data = new CaveSettingsData { hudBottom = hudPosition == 0, reducedMotion = reducedMotion == 1,
                masterVolume = Mathf.RoundToInt(volume * 100), muted = muted == 1 }; data.Sanitize(); return data;
        }
        public static CaveSettingsData Parse(string json)
        {
            try { var data = JsonUtility.FromJson<CaveSettingsData>(json) ?? new CaveSettingsData(); data.Sanitize(); return data; }
            catch (ArgumentException) { return new CaveSettingsData(); }
        }
        public static void Change(Action<CaveSettingsData> edit, bool custom = true)
        {
            edit(Data); Data.Sanitize();
            if (custom) Data.preset = "Custom";
            ApplyFramePacing(); Save(); Changed?.Invoke();
        }
        public static void Reset()
        {
            // Display changes require the separate Keep/Revert transaction.
            var width = Data.width; var height = Data.height; var fullscreen = Data.fullscreen;
            Data = new CaveSettingsData { width = width, height = height, fullscreen = fullscreen };
            ApplyFramePacing(); Save(); Changed?.Invoke();
        }
        public static void Save()
        { if (!Temporary) { PlayerPrefs.SetString(Key, JsonUtility.ToJson(Data)); PlayerPrefs.Save(); } }
        public static void ApplyFramePacing()
        {
            if (CaveNativeDisplay.IsMacPlayer)
            {
                // Unity's VSync semaphore stalled this macOS player, even without
                // readbacks. Metal synchronizes the swap layer instead.
                QualitySettings.vSyncCount = 0;
                var refresh = Mathf.Clamp(Mathf.RoundToInt((float)UnityEngine.Screen.currentResolution.refreshRateRatio.value),30,360);
                Application.targetFrameRate = Data.vSync ? refresh : Data.frameCap == 0 ? -1 : Data.frameCap;
                CaveNativeDisplay.Apply(Data.vSync);
            }
            else
            {
                var sync=Data.vSync?1:0;
                if(QualitySettings.vSyncCount!=sync)QualitySettings.vSyncCount=sync;
                Application.targetFrameRate=Data.vSync||Data.frameCap==0?-1:Data.frameCap;
            }
        }
    }

    // Wall-clock presentation controls never change Python's 60 Hz step clock.
    public sealed class CavePresentation : MonoBehaviour
    {
        public bool Pending { get; private set; }
        public float SecondsLeft => Mathf.Max(0, deadline - Time.realtimeSinceStartup);
        public float Fps { get; private set; }
        int oldWidth, oldHeight, requestedWidth, requestedHeight, observedWidth, observedHeight;
        FullScreenMode oldMode;
        bool requestedFullscreen;
        float deadline, resizeSince, sampleTime;
        int sampleFrames;
        public void ApplyBoot()
        {
            if (!CaveSettings.Temporary)
            {
                var d = CaveSettings.Data;
                SetDisplay(d.width, d.height, d.fullscreen);
            }
            observedWidth = UnityEngine.Screen.width; observedHeight = UnityEngine.Screen.height;
        }
        public void ApplyDisplay(int width, int height, bool fullscreen)
        {
            if (Pending) Revert();
            oldWidth = UnityEngine.Screen.width; oldHeight = UnityEngine.Screen.height; oldMode = UnityEngine.Screen.fullScreenMode;
            if (CaveSettings.Temporary) Debug.Log("Display transaction saved fullscreen=" + UnityEngine.Screen.fullScreen + " mode=" + oldMode + " size=" + oldWidth + "x" + oldHeight);
            requestedWidth = Mathf.Clamp(width, 640, 7680); requestedHeight = Mathf.Clamp(height, 384, 4320);
            requestedFullscreen = fullscreen;
            Pending = true; deadline = Time.realtimeSinceStartup + 15;
            SetDisplay(requestedWidth, requestedHeight, fullscreen);
        }
        static void SetDisplay(int width, int height, bool fullscreen)
        {
            if (fullscreen)
            {
                var display = UnityEngine.Screen.mainWindowDisplayInfo;
                var resolution = UnityEngine.Screen.currentResolution;
                width = display.width > 0 ? display.width : resolution.width;
                height = display.height > 0 ? display.height : resolution.height;
            }
            UnityEngine.Screen.SetResolution(width, height, fullscreen ? FullScreenMode.FullScreenWindow : FullScreenMode.Windowed);
        }
        public void Keep()
        {
            if (!Pending) return;
            Pending = false;
            CaveSettings.Change(d => { d.width = requestedWidth; d.height = requestedHeight; d.fullscreen = requestedFullscreen; }, false);
            Observe();
        }
        public void Revert()
        {
            if (!Pending) return;
            if (CaveSettings.Temporary) Debug.Log("Display revert requested mode=" + oldMode + " focused=" + Application.isFocused + " native=" + CaveNativeDisplay.WindowState);
            Pending = false;
            UnityEngine.Screen.SetResolution(oldWidth, oldHeight, oldMode); Observe();
        }
        void Observe()
        { observedWidth = UnityEngine.Screen.width; observedHeight = UnityEngine.Screen.height; resizeSince = Time.realtimeSinceStartup; }
        void LateUpdate() { if(CaveNativeDisplay.IsMacPlayer) CaveNativeDisplay.Apply(CaveSettings.Data.vSync); }
        void Update()
        {
            sampleFrames++; sampleTime += Time.unscaledDeltaTime;
            if (sampleTime >= .5f) { Fps = sampleFrames / sampleTime; sampleFrames = 0; sampleTime = 0; }
            if (Pending) { if (SecondsLeft <= 0) Revert(); return; }
            if (CaveSettings.Temporary || UnityEngine.Screen.fullScreen) return;
            if (observedWidth != UnityEngine.Screen.width || observedHeight != UnityEngine.Screen.height) { Observe(); return; }
            var d = CaveSettings.Data;
            if (Time.realtimeSinceStartup - resizeSince > 1 && (d.width != observedWidth || d.height != observedHeight || d.fullscreen))
                CaveSettings.Change(value => { value.width = observedWidth; value.height = observedHeight; value.fullscreen = false; }, false);
        }
        void OnApplicationQuit() { Revert(); }
        void OnDisable() { Revert(); }
    }
}

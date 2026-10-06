using System;
using System.Collections;
using System.IO;
using UnityEngine;

namespace CrystalCaves.Pilot
{
    // Explicit CLI-only rendering of snapshots recorded from the real Python
    // simulation. This does not replace physics or write gameplay preferences.
    public sealed partial class CavePilot
    {
        [Serializable] sealed class ReplayFrames
        {
            public int fps;
            public CaveSnapshot[] frames;
        }

        [Serializable] sealed class ReplayCaptureReport
        {
            public bool success;
            public int frames, fps, width, height;
        }

        bool replayCapture;

        bool TryStartReplayCapture(string[] arguments)
        {
            string input = null, output = null;
            for (var index = 0; index + 1 < arguments.Length; index++)
            {
                if (arguments[index] == "--render-replay") input = arguments[index + 1];
                if (arguments[index] == "--replay-output") output = arguments[index + 1];
            }
            if (input == null) return false;
            var replay = JsonUtility.FromJson<ReplayFrames>(File.ReadAllText(input));
            Debug.Log("Cave capture: replay parsed");
            if (replay == null || replay.fps < 1 || replay.fps > 60 || replay.frames == null || replay.frames.Length == 0)
                throw new InvalidDataException("Replay requires 1–60 fps and recorded snapshots.");
            foreach (var frame in replay.frames)
                if (frame == null || frame.protocol != 1 || frame.player == null || frame.entities == null || frame.effects == null || frame.layout == null)
                    throw new InvalidDataException("Replay contains an incomplete game snapshot.");
            if (string.IsNullOrEmpty(output))
                throw new InvalidDataException("Specify --replay-output for the captured frames.");
            Directory.CreateDirectory(output);
            replayCapture = true;
            // CLI review images should show final scene colors immediately.
            UnityEngine.Rendering.SplashScreen.Stop(UnityEngine.Rendering.SplashScreen.StopBehavior.StopImmediate);
            HasExpedition = true;
            Screen = CaveScreen.Play;
            audioPlayer.SetPlaying(false); // The video mux uses the recorded event timeline.
            QualitySettings.vSyncCount = 0; // Deterministic readback cannot wait on a display refresh.
            Application.targetFrameRate = replay.fps;
            Time.captureFramerate = replay.fps;
            StartCoroutine(RenderReplay(replay, output));
            Debug.Log("Cave capture: replay rendering started");
            return true;
        }

        IEnumerator RenderReplay(ReplayFrames replay, string output)
        {
            for (var index = 0; index < replay.frames.Length; index++)
            {
                state = replay.frames[index];
                view.OnSnapshot(state);
                CaveViewport.Preview = Screen == CaveScreen.Options;
                postFx.enabled = CavePostFx.Needed;
                world.Apply(state);
                world.Animate(state, 1f / replay.fps);
                yield return new WaitForEndOfFrame();
                var texture = ScreenCapture.CaptureScreenshotAsTexture();
                try { File.WriteAllBytes(Path.Combine(output, "frame-" + index.ToString("00000") + ".png"), texture.EncodeToPNG()); }
                finally { Destroy(texture); }
            }
            var report = new ReplayCaptureReport
            {
                success = true, frames = replay.frames.Length, fps = replay.fps,
                width = UnityEngine.Screen.width, height = UnityEngine.Screen.height
            };
            File.WriteAllText(Path.Combine(output, "capture-report.json"), JsonUtility.ToJson(report, true));
            Debug.Log("Replay capture saved " + report.frames + " native frames to " + output);
            Time.captureFramerate = 0;
            Application.Quit(0);
        }
    }
}

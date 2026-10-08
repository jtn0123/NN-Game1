using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace CrystalCaves.Pilot
{
    public enum CaveScreen { Title, Play, Pause, Caves, Options, Lab, Result, Controls }

    public sealed partial class CavePilot : MonoBehaviour
    {
        CaveConnection connection;
        CaveWorld world;
        CaveAudio audioPlayer;
        CaveInterface view;
        CavePresentation presentation;
        CavePostFx postFx;
        public CavePresentation Presentation => presentation;
        Camera gameCamera;
        CaveSnapshot state;
        readonly Stack<CaveScreen> history = new Stack<CaveScreen>();
        bool wasConnected, resetting, pendingMine;
        int pendingCave = -1, savedEpisode = -1;
        float accumulator, heartbeat, inputUntil;
        bool queuedJump, queuedShoot, queuedInteract;
        string warning = "";
        string pendingMode;
        bool smoke, visualSmoke;
        bool mechanismSmoke, mechanismsChecked, originalHudBottom, thornSeen;
        bool referenceSmoke, referenceChecked, stalactiteSeen, enemyHitChecked, bonesChecked, projectilesChecked, pickupsChecked;
        int capturedTrapStep = -1;
        int capturedLiftStep = -1;
        int motionCapturedStep = -1, smokeAiStep;
        int smokeStage, smokeLevel, capturedLevel = -1;
        float smokeStart;
        string capturePath, reportPath;

        public CaveSnapshot State => state;
        public CaveScreen Screen { get; private set; } = CaveScreen.Title;
        public bool Connected => settingsSmoke || replayCapture || (connection != null && connection.Connected);
        public bool Busy => connection == null || connection.Busy || resetting || pendingCave >= 0 || pendingMine;
        public bool OpeningCave => resetting || pendingCave >= 0 || pendingMine;
        public bool HasExpedition { get; private set; }
        public bool LabVisible { get; private set; }
        public string Warning => warning;
        public CaveAudio Sound => audioPlayer;
        public int SelectedCave { get; private set; }
        public Camera GameCamera => gameCamera;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        static void Launch()
        {
            if (FindFirstObjectByType<CavePilot>() == null)
                new GameObject("Crystal Caves").AddComponent<CavePilot>();
        }

        void Awake()
        {
            var captureTrace = Array.IndexOf(Environment.GetCommandLineArgs(), "--render-replay") >= 0 || Array.IndexOf(Environment.GetCommandLineArgs(), "--settings-smoke") >= 0;
            if (captureTrace) Debug.Log("Cave capture: creating camera and world");
            var settingsArguments = Environment.GetCommandLineArgs();
            var temporarySettings = captureTrace || Array.IndexOf(settingsArguments, "--smoke-report") >= 0 || Array.IndexOf(settingsArguments, "--settings-smoke") >= 0;
            var settingsIndex = Array.IndexOf(settingsArguments, "--presentation-settings");
            var settingsJson = settingsIndex >= 0 && settingsIndex + 1 < settingsArguments.Length ? File.ReadAllText(settingsArguments[settingsIndex + 1]) : null;
            if (captureTrace) Debug.Log("Cave capture: loading settings");
            CaveSettings.Load(temporarySettings || settingsJson != null, settingsJson);
            if (captureTrace) Debug.Log("Cave capture: settings loaded");
            presentation = gameObject.AddComponent<CavePresentation>();
            presentation.ApplyBoot();
            Application.runInBackground = true;
            SelectedCave = Mathf.Clamp(PlayerPrefs.GetInt("cave-last-selected", 0), 0, 15);
            // Clear the complete framebuffer before the cropped cave camera.
            // Otherwise translucent menu plates retain pixels from the last page.
            var backdrop = new GameObject("Display letterbox", typeof(Camera)).GetComponent<Camera>();
            backdrop.depth = -1; backdrop.cullingMask = 0; backdrop.clearFlags = CameraClearFlags.SolidColor; backdrop.backgroundColor = Color.black;
            var cameraObject = new GameObject("Cave Camera", typeof(Camera), typeof(AudioListener));
            gameCamera = cameraObject.GetComponent<Camera>();
            postFx = cameraObject.AddComponent<CavePostFx>();
            postFx.enabled = CavePostFx.Needed;
            gameCamera.orthographic = true;
            gameCamera.orthographicSize = 8;
            gameCamera.backgroundColor = new Color(.035f, .06f, .07f);
            gameCamera.clearFlags = CameraClearFlags.SolidColor;
            gameCamera.transform.position = new Vector3(15, -15, -10);
            world = new CaveWorld(gameCamera);
            if (captureTrace) Debug.Log("Cave capture: loading audio");
            audioPlayer = new CaveAudio(gameObject);
            if (captureTrace) Debug.Log("Cave capture: loading interface");
            view = new CaveInterface(this);
            if (captureTrace) Debug.Log("Cave capture: loading preview");
            var preview = Resources.Load<TextAsset>("PilotPreview");
            if (preview != null) { state = JsonUtility.FromJson<CaveSnapshot>(preview.text); world.Apply(state); }
            var port = 8766;
            var arguments = Environment.GetCommandLineArgs();
            for (var index = 0; index < arguments.Length; index++)
            {
                if (arguments[index] == "--bridge-port" && index + 1 < arguments.Length) int.TryParse(arguments[index + 1], out port);
                if (arguments[index] == "--visual-smoke") visualSmoke = true;
                if (arguments[index] == "--mechanism-smoke") mechanismSmoke = true;
                if (arguments[index] == "--reference-smoke") referenceSmoke = true;
                if (arguments[index] == "--capture" && index + 1 < arguments.Length) capturePath = arguments[index + 1];
                if (arguments[index] == "--smoke-report" && index + 1 < arguments.Length) { smoke = true; reportPath = arguments[index + 1]; }
            }
            smokeStart = Time.realtimeSinceStartup;
            if (captureTrace) Debug.Log("Cave capture: loading replay");
            if (TryStartSettingsSmoke(arguments) || TryStartReplayCapture(arguments)) return;
            UnityEngine.Rendering.SplashScreen.Stop(UnityEngine.Rendering.SplashScreen.StopBehavior.StopImmediate);
            Debug.Log("Crystal Caves ready; connecting to local simulation on port " + port);
            connection = new CaveConnection(port);
        }

        void Update()
        {
            CaveViewport.Preview = Screen == CaveScreen.Options;
            postFx.enabled = CavePostFx.Needed;
            if (replayCapture || settingsSmoke) return;
            while (connection.Read(out var json))
            {
                if (!CaveSnapshotReader.TryRead(json, out var next))
                {
                    warning = "The cave could not be opened. Please try opening it again.";
                    resetting = pendingMine = HasExpedition = false;
                    pendingCave = -1; pendingMode = null; accumulator = 0;
                    queuedJump = queuedShoot = queuedInteract = false;
                    Screen = CaveScreen.Title; continue;
                }
                if (!string.IsNullOrEmpty(next.error))
                {
                    warning = next.error; Screen = CaveScreen.Pause;
                    resetting = pendingMine = false; pendingCave = -1; pendingMode = null;
                    accumulator = 0; queuedJump = queuedShoot = queuedInteract = false;
                    continue;
                }
                var changedEpisode = state == null || next.episode != state.episode;
                state = next;
                view.OnSnapshot(state);
                world.Apply(state);
                audioPlayer.Play(state.sounds);
                if (changedEpisode) accumulator = 0;
                if (resetting && changedEpisode)
                { resetting = false; HasExpedition = true; Screen = CaveScreen.Play; }
                if (state.realm == "mine" && state.portal_level >= 0 && HasExpedition && !OpeningCave) PlayCave(state.portal_level);
                if (state.done && HasExpedition)
                {
                    if (Screen == CaveScreen.Play || Screen == CaveScreen.Pause)
                    { Screen = CaveScreen.Result; LabVisible = false; }
                    if (!smoke && !CaveSettings.Temporary && state.won && state.human_only && savedEpisode != state.episode)
                    {
                        savedEpisode = state.episode;
                        PlayerPrefs.SetInt("cave-cleared-" + state.level, 1);
                        PlayerPrefs.SetInt("cave-best-" + state.level,
                            Mathf.Max(state.score, PlayerPrefs.GetInt("cave-best-" + state.level, 0)));
                        PlayerPrefs.Save();
                    }
                }
            }
            if (wasConnected && !Connected)
            { Home(); HasExpedition = false; resetting = false; pendingCave = -1; pendingMode = null; pendingMine = false; savedEpisode = -1; }
            wasConnected = Connected;
            if (!smoke) { HandleKeys(); BufferInput(); }
            if (pendingMine && Connected && !connection.Busy)
            {
                var cleared = new List<int>();
                for (var index = 0; index < 16; index++) if (PlayerPrefs.GetInt("cave-cleared-" + index, 0) == 1) cleared.Add(index);
                if (Send(new CaveCommand { op = "mine", cleared = cleared.ToArray() }))
                { pendingMine = false; resetting = true; }
            }
            else if (pendingCave >= 0 && Connected && !connection.Busy)
            {
                if (state.mode == "ai") Send(new CaveCommand { op = "mode", mode = "human" });
                else if (Send(new CaveCommand { op = "reset", level = pendingCave }))
                { pendingCave = -1; resetting = true; }
            }
            else if (pendingMode != null && Connected && !connection.Busy)
            { if (Send(new CaveCommand { op = "mode", mode = pendingMode })) pendingMode = null; }
            var playing = Connected && HasExpedition && Screen == CaveScreen.Play && !ScreenshotMode && !presentation.Pending && !pendingMine && pendingCave < 0 && pendingMode == null && !resetting && !state.done;
            if (playing && !smoke)
            {
                accumulator = Mathf.Min(accumulator + Time.unscaledDeltaTime, 8 / 60f);
                var count = Mathf.Min(8, Mathf.FloorToInt(accumulator * 60));
                if (count > 0)
                {
                    CaveCommand command;
                    if (state.mode == "human")
                    {
                        var controls = new CaveHumanControl[count];
                        var buttons = HumanControls();
                        for (var index = 0; index < count; index++) controls[index] = buttons;
                        command = new CaveCommand { op = "human_step", controls = controls };
                    }
                    else command = new CaveCommand { op = "step", actions = new int[count] };
                    if (Send(command))
                    { accumulator -= count / 60f; queuedJump = queuedShoot = queuedInteract = false; }
                }
            }
            else if (!smoke && !pendingMine && pendingCave < 0 && !resetting)
            {
                if (Screen != CaveScreen.Play) accumulator = 0;
                heartbeat += Time.unscaledDeltaTime;
                if (heartbeat > .5f && Send(new CaveCommand { op = "snapshot" })) heartbeat = 0;
            }
            world.Animate(state, ScreenshotMode ? 0 : Time.unscaledDeltaTime);
            audioPlayer.SetPlaying(Connected && HasExpedition && !ScreenshotMode && (Screen == CaveScreen.Play || Screen == CaveScreen.Result));
            if (smoke) SmokeUpdate();
        }

        void HandleKeys()
        {
            if (ScreenshotMode) { HandleScreenshotKeys(); return; }
            if (presentation.Pending) { view.HandleMenuInput(); return; }
            var controlsMenu = Screen == CaveScreen.Controls;
            if (view.HandleControlsInput()) return;
            var startedInPlay = Screen == CaveScreen.Play;
            if (!startedInPlay && !controlsMenu) view.HandleMenuInput();
            if (startedInPlay && (CaveControls.Pressed(CaveControl.Pause) || Input.GetKeyDown(KeyCode.JoystickButton7))) Back();
            if (Screen != CaveScreen.Play && !controlsMenu)
            {
                if (Input.GetKeyDown(KeyCode.C)) Open(CaveScreen.Caves);
                if (Input.GetKeyDown(KeyCode.O)) Open(CaveScreen.Options);
                if (Input.GetKeyDown(KeyCode.H)) Home();
            }
            if (Input.GetKeyDown(KeyCode.M)) audioPlayer.ToggleMute();
            if (Input.GetKeyDown(KeyCode.F11)) presentation.ApplyDisplay(CaveSettings.Data.width, CaveSettings.Data.height, !UnityEngine.Screen.fullScreen);
            if (Input.GetKeyDown(KeyCode.F2))
            {
                if (Screen == CaveScreen.Play) LabVisible = !LabVisible;
                else if (Screen == CaveScreen.Lab) Back();
                else Open(CaveScreen.Lab);
            }
            if (Input.GetKeyDown(KeyCode.F12))
            {
                if (!string.IsNullOrEmpty(capturePath)) StartCoroutine(CaptureReview());
                else SaveScreenshot();
            }
        }

        void BufferInput()
        {
            if (Screen != CaveScreen.Play || ScreenshotMode || presentation.Pending || !Application.isFocused || state.mode != "human" || Time.unscaledTime > inputUntil)
                queuedJump = queuedShoot = queuedInteract = false;
            if (Screen != CaveScreen.Play || ScreenshotMode || presentation.Pending || !Application.isFocused || state.mode != "human") return;
            var jump = CaveControls.Pressed(CaveControl.Jump) || Input.GetKeyDown(KeyCode.JoystickButton0);
            var shoot = CaveControls.Pressed(CaveControl.Shoot) || Input.GetKeyDown(KeyCode.JoystickButton2);
            var interact = CaveControls.Pressed(CaveControl.Interact) || Input.GetKeyDown(KeyCode.JoystickButton3) || state.realm == "mine" && Input.GetKeyDown(KeyCode.Return);
            if (jump || shoot || interact)
            { queuedJump |= jump; queuedShoot |= shoot; queuedInteract |= interact; inputUntil = Time.unscaledTime + .15f; }
        }

        CaveHumanControl HumanControls()
        {
            if (!Application.isFocused) return new CaveHumanControl();
            var left = CaveControls.Held(CaveControl.MoveLeft) || Input.GetAxisRaw("CaveHorizontal") < -.55f;
            var right = CaveControls.Held(CaveControl.MoveRight) || Input.GetAxisRaw("CaveHorizontal") > .55f;
            return new CaveHumanControl
            {
                move = left == right ? 0 : left ? -1 : 1,
                jump = queuedJump || CaveControls.Held(CaveControl.Jump) || Input.GetKey(KeyCode.JoystickButton0) || Input.GetAxisRaw("CaveVertical") > .55f,
                shoot = queuedShoot || CaveControls.Held(CaveControl.Shoot) || Input.GetKey(KeyCode.JoystickButton2),
                interact = queuedInteract || CaveControls.Held(CaveControl.Interact) || Input.GetKey(KeyCode.JoystickButton3) || Input.GetAxisRaw("CaveVertical") < -.55f || state.realm == "mine" && Input.GetKey(KeyCode.Return)
            };
        }

        bool Send(CaveCommand command) => connection.Send(JsonUtility.ToJson(command));
        public void Select(int cave) { SelectedCave = Mathf.Clamp(cave, 0, 15); }
        public void Play()
        {
            if (HasExpedition && !state.done) Resume();
            else OpenMine();
        }
        public void OpenMine()
        {
            if (!Connected || OpeningCave) return;
            LabVisible = false; history.Clear(); warning = ""; accumulator = 0;
            Screen = CaveScreen.Title; pendingMode = null; pendingCave = -1; pendingMine = true;
        }
        public void PlayCave(int cave)
        {
            if (!Connected || OpeningCave) return;
            Select(cave);
            if (!CaveSettings.Temporary) PlayerPrefs.SetInt("cave-last-selected", SelectedCave);
            LabVisible = false;
            history.Clear();
            warning = "";
            accumulator = 0;
            Screen = CaveScreen.Title;
            pendingCave = SelectedCave;
            pendingMode = null;
        }
        public void Open(CaveScreen screen)
        { if (screen != Screen) { history.Push(Screen); Screen = screen; accumulator = 0; } }
        public void Home() { Screen = CaveScreen.Title; LabVisible = false; history.Clear(); accumulator = 0; }
        public void Resume() { if (Connected && HasExpedition && !state.done) Screen = CaveScreen.Play; }
        public void Back()
        {
            if (presentation.Pending) { presentation.Revert(); return; }
            if (ScreenshotMode) { ExitScreenshotMode(); return; }
            if (Screen == CaveScreen.Play) { Screen = CaveScreen.Pause; LabVisible = false; }
            else if (Screen == CaveScreen.Pause) Resume();
            else if (Screen == CaveScreen.Result) Home();
            else if (Screen != CaveScreen.Title && Screen != CaveScreen.Result) Screen = history.Count > 0 ? history.Pop() : CaveScreen.Title;
            accumulator = 0;
        }
        public void WatchAgent()
        {
            if (!Connected || OpeningCave || !state.ai_available || state.done) return;
            pendingMode = "ai";
            history.Clear(); Screen = CaveScreen.Play; HasExpedition = true; LabVisible = true;
        }
        public void TakeControl()
        {
            if (!Connected || OpeningCave) return;
            pendingMode = "human";
            LabVisible = false; HasExpedition = true; Screen = state.done ? CaveScreen.Result : CaveScreen.Pause;
        }
        public void CloseLab() { if (Screen == CaveScreen.Lab) Back(); else LabVisible = false; }

        void OnGUI() { view.Draw(); }
        void OnApplicationFocus(bool focus)
        { if (!focus && !smoke && !replayCapture && !settingsSmoke && Screen == CaveScreen.Play) { Screen = CaveScreen.Pause; accumulator = 0; } }
        void OnDestroy() { connection?.Dispose(); CaveViewport.Preview = CaveViewport.HideHudStrip = false; CaveVisualClock.Frozen = false; }

        // Opt-in native smoke checks the actual connection, policy and menu renders.
        void SmokeUpdate()
        {
            if (Time.realtimeSinceStartup - smokeStart > 120) { FinishSmoke(false, "Bridge smoke timed out"); return; }
            if (!Connected || connection.Busy || state == null) return;
            if (smokeStage == 0) { if (Send(new CaveCommand { op = "reset", level = 0 })) smokeStage = 1; }
            else if (smokeStage == 1 && state.steps == 0)
            { if (Send(new CaveCommand { op = "step", actions = new[] { 2, 2, 2, 2, 2, 2, 2, 6 } })) smokeStage = 2; }
            else if (smokeStage == 2 && state.steps == 8)
            {
                if (state.sounds == null || Array.IndexOf(state.sounds, "shoot") < 0) { FinishSmoke(false, "Human fire feedback missing"); return; }
                smokeStage = 20; StartCoroutine(CaptureCombat());
            }
            else if (smokeStage == 21)
            {
                if (visualSmoke) { HasExpedition = true; Screen = CaveScreen.Play; smokeStage = 22; }
                else StartAgentSmoke();
            }
            else if (smokeStage == 22)
            {
                if (state.done || state.steps >= 128)
                { if (Send(new CaveCommand { op = "reset", level = 0 })) smokeStage = 24; }
                else if (state.steps % 4 == 0 && motionCapturedStep != state.steps)
                { motionCapturedStep = state.steps; smokeStage = 23; StartCoroutine(CaptureMotion()); }
                else Send(new CaveCommand { op = "step", actions = new[] { state.steps < 34 ? 2 : state.steps < 76 ? 5 : 2 } });
            }
            else if (smokeStage == 24 && state.steps == 0) StartAgentSmoke();
            else if (smokeStage == 3 && state.mode == "ai")
            { smokeAiStep = state.steps + 4; if (Send(new CaveCommand { op = "step", actions = new[] { 0, 0, 0, 0 } })) smokeStage = 4; }
            else if (smokeStage == 4 && state.steps == smokeAiStep)
            {
                if (state.q_values == null || state.q_values.Length != 10) { FinishSmoke(false, "AI returned no action values"); return; }
                smokeStage = 40;
                StartCoroutine(CaptureAgent());
            }
            else if (smokeStage == 41)
            { TakeControl(); smokeStage = 5; }
            else if (smokeStage == 5 && state.mode == "human")
            { smokeLevel = 1; if (Send(new CaveCommand { op = "reset", level = smokeLevel })) smokeStage = 6; }
            else if (smokeStage == 6 && state.level == smokeLevel && state.steps == 0)
            {
                if ((referenceSmoke || smokeLevel == 1 || smokeLevel == 2 || smokeLevel == 4 || smokeLevel == 6 || smokeLevel == 7 || smokeLevel == 8 || smokeLevel == 13 || smokeLevel == 15) && capturedLevel != smokeLevel)
                { capturedLevel = smokeLevel; smokeStage = 60; StartCoroutine(CaptureCaveTheme()); }
                else if (smokeLevel < state.levels.Length - 1)
                { if (Send(new CaveCommand { op = "reset", level = smokeLevel + 1 })) smokeLevel++; }
                else if (Send(new CaveCommand { op = "reset", level = 0 })) smokeStage = 7;
            }
            else if (smokeStage == 7 && state.level == 0 && state.steps == 0)
            { smokeStage = 8; StartCoroutine(CaptureSmoke()); }
            else if (smokeStage == 9)
            {
                if (!state.done && state.steps > 1200) FinishSmoke(false, "Ground hazard did not end the run");
                else if (!state.done)
                {
                    var action = SmokeHazardAction();
                    Send(new CaveCommand { op = "step", actions = new[] { action, action, action, action, action, action, action, action } });
                }
                else if (Screen != CaveScreen.Result) FinishSmoke(false, "Completed run did not open results");
                else { smokeStage = 10; StartCoroutine(CaptureResult()); }
            }
            else if (smokeStage == 70 && state.realm == "mine" && !Busy)
            { smokeStage = 71; StartCoroutine(CaptureMine()); }
            else if (smokeStage == 72 && state.realm == "mine")
            {
                if (state.near_entrance == 0)
                { smokeStage = 73; StartCoroutine(EnterMineDoor()); }
                else if (state.steps > 90) FinishSmoke(false, "Main mine entrance was not reachable by walking");
                else Send(new CaveCommand { op = "step", actions = new[] { 2, 2, 2, 2 } });
            }
            else if (smokeStage == 74 && state.realm == "cave" && state.level == 0 && !Busy)
            { OpenMine(); smokeStage = 75; }
            else if (smokeStage == 75 && state.realm == "mine" && !Busy)
            {
                if (state.near_entrance != 0 || state.human_only || state.done || state.levels.Length != 16)
                    FinishSmoke(false, "Returning to the main mine lost its doorway or session boundaries");
                else FinishSmoke(true, "Human play, AI and takeover, sixteen caves, menus, game-over, restart, completion fixtures, playable main mine and real doorway entry/return passed");
            }
            else if (smokeStage == 11 && state.level == 3 && state.steps == 0 && !Busy)
            {
                if (Screen != CaveScreen.Play || state.mode != "human") FinishSmoke(false, "Restart did not return human play");
                else { smokeStage = 12; StartCoroutine(CaptureCompletionFixture()); }
            }
            else if (smokeStage == 80 && state.level == 2 && state.steps == 0)
            { HasExpedition = true; Screen = CaveScreen.Play; smokeStage = 81; }
            else if (smokeStage == 81)
            {
                if (state.health != 3 || state.done) { FinishSmoke(false, "Real-input lift boarding lost health"); return; }
                foreach (var entity in state.entities)
                    if (entity.id.StartsWith("thorn_") && entity.sprite == "green_thorn_4") thornSeen = true;
                if (state.steps >= 560 && capturedLiftStep < 0)
                { capturedLiftStep = state.steps; smokeStage = 82; StartCoroutine(CaptureMechanisms()); }
                else if (state.steps >= 840)
                {
                    var lift = Array.Find(state.entities, entity => entity.id == "lift_0");
                    if (!thornSeen || lift == null || !state.player.grounded || state.player.climbing ||
                        Mathf.Abs(state.player.y + 30 - Mathf.Floor(lift.y)) > 1 || state.player.y >= 300)
                    { FinishSmoke(false, "Lift ride or proximity thorn state did not match the authoritative game"); return; }
                    smokeStage = 83; StartCoroutine(CaptureHudPositions());
                }
                else
                {
                    var actions = new int[Mathf.Min(8, 840 - state.steps)];
                    for (var index = 0; index < actions.Length; index++)
                    { var step = state.steps + index; actions[index] = step < 470 ? 0 : step < 500 ? 2 : step < 514 ? 5 : 0; }
                    Send(new CaveCommand { op = "step", actions = actions });
                }
            }
            else if (smokeStage == 84)
            { mechanismsChecked = true; StartAgentSmoke(); }
            else if (smokeStage == 90 && state.level == 4 && state.steps == 0)
            { HasExpedition = true; Screen = CaveScreen.Play; smokeStage = 91; }
            else if (smokeStage == 91)
            {
                if (state.health != 3 || state.done) { FinishSmoke(false, "Real-input stalactite dodge lost health"); return; }
                var trap = Array.Find(state.entities, entity => entity.id == "stalactite_0");
                if (trap != null && trap.y > 544) stalactiteSeen = true;
                if (capturedTrapStep != state.steps)
                { capturedTrapStep = state.steps; smokeStage = 92; StartCoroutine(CaptureStalactite()); }
                else if (state.steps >= 196)
                {
                    if (!stalactiteSeen || trap != null) { FinishSmoke(false, "Ceiling trap did not release and break on the platform"); return; }
                    referenceChecked = true; StartAgentSmoke();
                }
                else
                {
                    var actions = new int[Mathf.Min(4, 196 - state.steps)];
                    for (var index = 0; index < actions.Length; index++)
                    { var step = state.steps + index; actions[index] = step < 10 ? 0 : step < 25 ? 5 : step < 134 ? 2 : step < 154 ? 1 : 0; }
                    Send(new CaveCommand { op = "step", actions = actions });
                }
            }
        }

        IEnumerator CaptureStalactite()
        {
            yield return null;
            yield return new WaitForEndOfFrame();
            CaptureNamed("trap-motion-" + state.steps.ToString("000"));
            File.WriteAllText(Path.Combine(Path.GetDirectoryName(capturePath), "trap-motion-" + state.steps.ToString("000") + ".json"), JsonUtility.ToJson(state, true));
            smokeStage = 91;
        }

        IEnumerator CaptureMechanisms()
        {
            yield return null;
            yield return new WaitForEndOfFrame(); CaptureNamed("lift-and-thorn");
            File.WriteAllText(Path.Combine(Path.GetDirectoryName(capturePath), "lift-and-thorn.json"), JsonUtility.ToJson(state, true));
            smokeStage = 81;
        }

        IEnumerator CaptureHudPositions()
        {
            if (!CaveVisualSettings.HudBottom) CaveVisualSettings.ToggleHud();
            yield return null;
            yield return new WaitForEndOfFrame(); CaptureNamed("hud-bottom");
            var bottom = gameCamera.pixelRect;
            CaveVisualSettings.ToggleHud();
            yield return null;
            yield return new WaitForEndOfFrame(); CaptureNamed("hud-top");
            var top = gameCamera.pixelRect;
            if (top.width != bottom.width || top.height != bottom.height || bottom.y - top.y != 32 * CaveViewport.Scale)
            { FinishSmoke(false, "HUD placement did not reserve the matching camera strip"); yield break; }
            var texture = ScreenCapture.CaptureScreenshotAsTexture();
            var visible = false;
            for (var x = (int)top.x + 32; x < top.xMax - 32; x += 32)
            {
                var pixel = texture.GetPixel(x, (int)top.y + 24);
                visible |= pixel.r + pixel.g + pixel.b > .03f;
            }
            Destroy(texture);
            if (!visible) { FinishSmoke(false, "Top HUD left an obsolete black footer over the cave"); yield break; }
            Screen = CaveScreen.Options;
            yield return new WaitForEndOfFrame(); CaptureNamed("hud-options");
            if (CaveVisualSettings.HudBottom != originalHudBottom) CaveVisualSettings.ToggleHud();
            Screen = CaveScreen.Play; smokeStage = 84;
        }

        IEnumerator CaptureMine()
        {
            Screen = CaveScreen.Play;
            yield return null;
            yield return new WaitForEndOfFrame(); CaptureNamed("main-mine");
            Screen = CaveScreen.Pause;
            yield return new WaitForEndOfFrame(); CaptureNamed("mine-pause");
            Screen = CaveScreen.Play; smokeStage = 72;
        }
        IEnumerator EnterMineDoor()
        {
            yield return null;
            yield return new WaitForEndOfFrame(); CaptureNamed("mine-doorway");
            if (Send(new CaveCommand { op = "step", actions = new[] { 9 } })) smokeStage = 74;
            else { smokeStage = 72; }
        }

        int SmokeHazardAction()
        {
            // Reach the ground spikes with real inputs, then wait for actual damage.
            // Human play has no training timeout, so idling is no longer a death test.
            var target = float.PositiveInfinity;
            for (var row = 0; row < state.layout.Length; row++)
                for (var col = 0; col < state.layout[row].Length; col++)
                    if (state.layout[row][col] == '^' && Mathf.Abs(row * 32 - state.player.y) < 48)
                        target = Mathf.Min(target, col * 32 + 16);
            if (float.IsPositiveInfinity(target)) return 2;
            var center = state.player.x + 11;
            return center < target - 3 ? 2 : center > target + 3 ? 1 : 0;
        }

        void StartAgentSmoke()
        {
            if (mechanismSmoke && !mechanismsChecked)
            {
                originalHudBottom = CaveVisualSettings.HudBottom;
                if (Send(new CaveCommand { op = "reset", level = 2 })) smokeStage = 80;
                return;
            }
            if (referenceSmoke && !referenceChecked)
            {
                if (Send(new CaveCommand { op = "reset", level = 4 })) smokeStage = 90;
                return;
            }
            if (state.ai_available) { WatchAgent(); smokeStage = 3; } else smokeStage = 5;
        }
        IEnumerator CaptureMotion()
        {
            yield return null;
            yield return new WaitForEndOfFrame(); CaptureNamed("motion-" + state.steps.ToString("000")); smokeStage = 22;
        }
        // A presentation fixture, explicitly distinct from a completed gameplay route.
        // It does not touch Python, achievements, demos, or the authoritative snapshot.
        IEnumerator CaptureCompletionFixture()
        {
            var actual = state;
            state = JsonUtility.FromJson<CaveSnapshot>(JsonUtility.ToJson(actual));
            state.won = state.done = false; state.crystals = 0; state.exit_unlocked = true; state.human_only = false;
            state.entities = Array.FindAll(state.entities, entity => !entity.id.StartsWith("crystal_"));
            foreach (var entity in state.entities) if (entity.id == "exit") entity.sprite = entity.h > 32 ? "exit_open_tall" : "exit_open";
            world.Apply(state); Screen = CaveScreen.Play;
            yield return null;
            yield return new WaitForEndOfFrame(); CaptureNamed("all-gems-fixture");
            state.won = state.done = true; state.crystals = 0; state.score = 3200; state.human_only = false;
            Screen = CaveScreen.Result;
            yield return new WaitForSecondsRealtime(1.1f);
            yield return null;
            yield return new WaitForEndOfFrame(); CaptureNamed("completion-fixture");
            state = actual; world.Apply(state); Screen = CaveScreen.Play;
            OpenMine(); smokeStage = 70;
        }

        IEnumerator CaptureCombat()
        {
            HasExpedition = true; Screen = CaveScreen.Play;
            // Let the native startup fade finish before reviewing scene colors.
            yield return new WaitForSecondsRealtime(2);
            audioPlayer.Play(new[] { "gem" });
            yield return null;
            if (audioPlayer.CurrentSound != "gem" || !audioPlayer.IsPlaying)
            { FinishSmoke(false, "Classic crystal cue did not play"); yield break; }
            audioPlayer.Play(new[] { "shoot" });
            if (audioPlayer.CurrentSound != "gem")
            { FinishSmoke(false, "Lower-priority cue interrupted the speaker"); yield break; }
            audioPlayer.Play(new[] { "switch" });
            if (audioPlayer.CurrentSound != "switch")
            { FinishSmoke(false, "Higher-priority cue failed to replace the speaker"); yield break; }
            yield return new WaitForEndOfFrame(); CaptureNamed("combat");
            if (referenceSmoke)
            {
                yield return CaptureProjectilePresentation();
                if (!smoke) yield break;
                yield return CaptureGravityPresentation();
                if (!smoke) yield break;
            }
            HasExpedition = false; Screen = CaveScreen.Title; smokeStage = 21;
        }

        IEnumerator CaptureGravityPresentation()
        {
            var actual = state;
            if (actual.player.gravity_dir != 1)
            { FinishSmoke(false, "Normal gravity metadata missing from live bridge"); yield break; }
            state = JsonUtility.FromJson<CaveSnapshot>(JsonUtility.ToJson(actual));
            state.sounds = new string[0]; state.effects = new CaveEffect[0];
            state.player.invulnerable = false; state.player.climbing = false;
            var player = Array.Find(state.entities, entity => entity.id == "player");
            var renderer = GameObject.Find("player").GetComponent<SpriteRenderer>();
            foreach (var sign in new[] { 1, -1, 0 })
            {
                var gravity = sign < 0 ? -1 : 1;
                state.player.gravity_dir = sign; state.player.grounded = false;
                state.player.vx = 0; state.steps = 120; player.sprite = "mylo_idle";
                foreach (var facing in new[] { 1, -1 })
                {
                    player.flip = facing < 0; state.player.facing = facing;
                    foreach (var rising in new[] { true, false })
                    {
                        state.player.vy = (rising ? -3 : 3) * gravity;
                        world.Apply(state); world.Animate(state, 0);
                        var pose = rising ? "mylo_jump" : "mylo_fall";
                        if (renderer.flipY != (gravity < 0) || renderer.flipX != player.flip || renderer.sprite.texture.name != pose)
                        { FinishSmoke(false, "Gravity-relative orientation or airborne pose failed"); yield break; }
                    }
                }
                state.player.vy = 0; state.player.grounded = true; state.steps = 0;
                player.flip = false; state.player.facing = 1;
                player.x = actual.player.x - 1; player.y = actual.player.y - 2;
                if (gravity < 0)
                {
                    var col = Mathf.Clamp(Mathf.FloorToInt(actual.player.x / 32), 0, state.cols - 1);
                    var row = Mathf.Clamp(Mathf.FloorToInt(actual.player.y / 32) - 1, 0, state.rows - 1);
                    while (row > 0 && state.layout[row][col] != '#') row--;
                    player.y = (row + 1) * 32;
                }
                state.player.y = player.y + 2;
                world.Apply(state); world.Animate(state, 0);
                var shadow = GameObject.Find("player · contact shadow").GetComponent<SpriteRenderer>();
                var contactY = gravity < 0 ? player.y + 1 : player.y + player.h - 1;
                var expectedShadow = CaveWorld.Position(player.x + player.w / 2, contactY);
                if (!shadow.gameObject.activeSelf || Vector3.Distance(shadow.transform.position, expectedShadow) > .04f)
                { FinishSmoke(false, "Gravity contact shadow did not follow the supporting surface"); yield break; }
                foreach (var entity in state.entities)
                {
                    if (entity.id.StartsWith("enemy_") && GameObject.Find(entity.id).GetComponent<SpriteRenderer>().flipY)
                    { FinishSmoke(false, "Player gravity flipped an enemy"); yield break; }
                }
                yield return null;
                yield return new WaitForEndOfFrame();
                if (sign != 0)
                {
                    var suffix = sign < 0 ? "gravity-inverted-fixture" : "gravity-normal-fixture";
                    CaptureNamed(suffix);
                    File.WriteAllText(Path.Combine(Path.GetDirectoryName(capturePath), suffix + ".json"), JsonUtility.ToJson(state, true));
                }
            }
            state = actual; world.Apply(state); world.Animate(state, 0);
            if (renderer.flipY)
            { FinishSmoke(false, "Restoring normal live state retained inverted artwork"); yield break; }
        }
        IEnumerator CaptureCaveTheme()
        {
            Screen = CaveScreen.Play;
            yield return null;
            if (referenceSmoke)
            {
                foreach (var entity in state.entities)
                {
                    if (!entity.id.StartsWith("crystal_") && !entity.id.StartsWith("power_")) continue;
                    var expected = CaveWorld.Position(entity.x + entity.w / 2, entity.y + entity.h / 2);
                    expected = new Vector3(Mathf.Round(expected.x * 32) / 32, Mathf.Round(expected.y * 32) / 32, 0);
                    var renderer = GameObject.Find(entity.id).GetComponent<SpriteRenderer>();
                    if (Vector3.Distance(renderer.transform.position, expected) > .005f)
                    { FinishSmoke(false, "Pickup animation drifted away from its authored site"); yield break; }
                    pickupsChecked = true;
                }
            }
            yield return new WaitForEndOfFrame(); CaptureNamed("cave-" + (smokeLevel + 1).ToString("00"));
            if (referenceSmoke && smokeLevel == 1)
            {
                yield return CaptureExitPresentation();
                if (!smoke) yield break;
            }
            if (referenceSmoke && !enemyHitChecked && Array.Exists(state.entities, entity => entity.sprite == "dinosaur_enemy"))
            {
                yield return CaptureEnemyHitPresentation();
                if (!smoke) yield break;
                yield return CaptureBonesPresentation();
                if (!smoke) yield break;
            }
            Screen = CaveScreen.Pause; smokeStage = 6;
        }

        IEnumerator CaptureEnemyHitPresentation()
        {
            var actual = state;
            state = JsonUtility.FromJson<CaveSnapshot>(JsonUtility.ToJson(actual));
            state.sounds = new string[0]; state.effects = new CaveEffect[0];
            var enemy = Array.Find(state.entities, entity => entity.sprite == "dinosaur_enemy");
            var player = Array.Find(state.entities, entity => entity.id == "player");
            // Reframe an authored creature for presentation review. Real bullet
            // damage and expiry are covered separately by bridge regressions.
            state.player.x = enemy.x - 48; state.player.y = enemy.y + enemy.h - 30;
            state.player.vx = state.player.vy = 0; state.player.invulnerable = false;
            state.player.grounded = true; state.player.climbing = false;
            player.x = state.player.x - 1; player.y = state.player.y - 2;
            foreach (var hit in new[] { false, true })
            {
                enemy.hit = hit;
                for (var pose = 0; pose < 4; pose++)
                {
                    state.steps = (pose + 1) * 6; state.freeze_timer = 0;
                    world.Apply(state); world.Animate(state, 0);
                    var renderer = GameObject.Find(enemy.id).GetComponent<SpriteRenderer>();
                    var frame = state.steps / 6 % 4;
                    var expected = "dinosaur_enemy" + (hit ? "_hit" : "") + "_" + frame;
                    if (renderer.sprite.texture.name != expected || renderer.sprite.texture.width != 24 || renderer.sprite.texture.height != 64 || renderer.flipX != enemy.flip || renderer.flipY)
                    { FinishSmoke(false, "Enemy hit artwork lost its pose, dimensions or orientation"); yield break; }
                }
                yield return null;
                yield return new WaitForEndOfFrame();
                CaptureNamed(hit ? "enemy-hit-fixture" : "enemy-normal-fixture");
            }
            state = actual; world.Apply(state); world.Animate(state, 0);
            var restored = Array.Find(state.entities, entity => entity.id == enemy.id);
            var restoredRenderer = GameObject.Find(enemy.id).GetComponent<SpriteRenderer>();
            var restoredFrame = (state.steps + state.freeze_timer) / 6 % 4;
            var restoredName = "dinosaur_enemy" + (restored.hit ? "_hit" : "") + "_" + restoredFrame;
            if (restoredRenderer.sprite.texture.name != restoredName || restoredRenderer.flipX != restored.flip || restoredRenderer.flipY)
            { FinishSmoke(false, "Restoring the live creature retained its fixture hit pose"); yield break; }
            enemyHitChecked = true;
        }

        IEnumerator CaptureBonesPresentation()
        {
            var actual = state;
            state = JsonUtility.FromJson<CaveSnapshot>(JsonUtility.ToJson(actual));
            var enemy = Array.Find(state.entities, entity => entity.sprite == "dinosaur_enemy");
            var player = Array.Find(state.entities, entity => entity.id == "player");
            state.player.x = enemy.x - 48; state.player.y = enemy.y + enemy.h - 30;
            state.player.vx = state.player.vy = 0; state.player.invulnerable = false;
            state.player.grounded = true; state.player.climbing = false;
            player.x = state.player.x - 1; player.y = state.player.y - 2;
            state.entities = Array.FindAll(state.entities, entity => entity.id != enemy.id);
            state.sounds = new string[0];
            var effect = new CaveEffect { id = "review-bones", kind = "bones", text = "+200", x = enemy.x + enemy.w / 2 + .375f, y = enemy.y + enemy.h / 2 + .375f, max_ttl = 72 };
            state.effects = new[] { effect };
            foreach (var facing in new[] { 1, -1 })
            {
                effect.facing = facing;
                for (var phase = 0; phase < 4; phase++)
                {
                    effect.ttl = 72 - phase * 18;
                    world.Apply(state); world.Animate(state, 0);
                    var renderer = GameObject.Find("Feedback · bones").GetComponent<SpriteRenderer>();
                    var displayed = phase;
                    var offsets = new[] { 0, 1, 3, 6 };
                    var position = CaveWorld.Position(effect.x + displayed * 4 * facing, effect.y + offsets[displayed]);
                    position = new Vector3(Mathf.Round(position.x * 32) / 32, Mathf.Round(position.y * 32) / 32, 0);
                    if (renderer.sprite.texture.name != "defeat_bones_" + displayed || renderer.sprite.texture.width != 32 || renderer.sprite.texture.height != 32 || renderer.flipX != (facing < 0) || renderer.color.a != 1 || renderer.transform.localScale != Vector3.one || Vector3.Distance(renderer.transform.position, position) > .01f)
                    { FinishSmoke(false, "Bone breakup lost its discrete pose, drift, full scale or hard alpha"); yield break; }
                    yield return null;
                    yield return new WaitForEndOfFrame();
                    if (facing > 0) CaptureNamed("bones-" + phase + "-fixture");
                }
            }
            state.effects = new CaveEffect[0]; world.Apply(state);
            yield return null;
            if (GameObject.Find("Feedback · bones") != null)
            { FinishSmoke(false, "Expired bone feedback lingered in the native scene"); yield break; }
            state = actual; world.Apply(state); world.Animate(state, 0);
            if (!GameObject.Find(enemy.id).activeSelf)
            { FinishSmoke(false, "Bone fixture did not restore the live creature"); yield break; }
            bonesChecked = true;
        }

        IEnumerator CaptureProjectilePresentation()
        {
            var actual = state;
            state = JsonUtility.FromJson<CaveSnapshot>(JsonUtility.ToJson(actual));
            var entities = new List<CaveEntity>(Array.FindAll(state.entities, entity => !entity.id.StartsWith("bullet_")));
            var bullet = new CaveEntity { id = "bullet_review", sprite = "bullet", w = 10, h = 4 };
            entities.Add(bullet); state.entities = entities.ToArray();
            state.effects = new CaveEffect[0]; state.sounds = new string[0];
            state.player.invulnerable = state.player.climbing = false;
            var player = Array.Find(state.entities, entity => entity.id == "player");
            player.sprite = "mylo_shoot";
            foreach (var facing in new[] { 1, -1 })
            {
                state.player.facing = facing; player.flip = facing < 0; bullet.flip = facing < 0;
                bullet.x = actual.player.x + (facing > 0 ? 40 : -50) + .375f;
                bullet.y = actual.player.y + 12.375f;
                for (var frame = 0; frame < 4; frame++)
                {
                    state.steps = 12 + frame * 3;
                    world.Apply(state); world.Animate(state, 0);
                    var renderer = GameObject.Find(bullet.id).GetComponent<SpriteRenderer>();
                    var expected = CaveWorld.Position(bullet.x + 5, bullet.y + 2);
                    expected = new Vector3(Mathf.Round(expected.x * 32) / 32, Mathf.Round(expected.y * 32) / 32, 0);
                    if (renderer.sprite.texture.name != "bullet_" + frame || renderer.sprite.texture.width != 16 || renderer.sprite.texture.height != 8 || renderer.flipX != bullet.flip || Vector3.Distance(renderer.transform.position, expected) > .005f)
                    { FinishSmoke(false, "Capsule artwork lost its discrete pose, facing or collider-center anchor"); yield break; }
                    yield return null;
                    yield return new WaitForEndOfFrame();
                    if (facing > 0) CaptureNamed("projectile-" + frame + "-fixture");
                }
            }
            state = actual; world.Apply(state); world.Animate(state, 0);
            if (GameObject.Find(bullet.id) != null)
            { FinishSmoke(false, "Restoring live play left a projectile fixture visible"); yield break; }
            projectilesChecked = true;
        }

        IEnumerator CaptureExitPresentation()
        {
            var actual = state;
            state = JsonUtility.FromJson<CaveSnapshot>(JsonUtility.ToJson(actual));
            var exit = Array.Find(state.entities, entity => entity.id == "exit");
            var player = Array.Find(state.entities, entity => entity.id == "player");
            if (exit.h != 64 || exit.w != 32)
            { FinishSmoke(false, "Clear-headroom exit did not receive its tall presentation"); yield break; }
            // Reframe real authored machinery for art inspection; this does not
            // navigate a route or unlock the authoritative simulation.
            state.player.x = exit.x - 32; state.player.y = exit.y + exit.h - 30;
            player.x = state.player.x - 1; player.y = state.player.y - 2;
            foreach (var opened in new[] { false, true })
            {
                exit.sprite = opened ? "exit_open_tall" : "exit_locked_tall";
                state.exit_unlocked = opened;
                world.Apply(state); world.Animate(state, 0);
                var renderer = GameObject.Find("exit").GetComponent<SpriteRenderer>();
                if (renderer.sprite.texture.width != 32 || renderer.sprite.texture.height != 64 || renderer.sprite.texture.name != exit.sprite)
                { FinishSmoke(false, "Tall exit artwork or state failed in the native renderer"); yield break; }
                yield return null;
                yield return new WaitForEndOfFrame();
                CaptureNamed(opened ? "exit-open-fixture" : "exit-closed-fixture");
            }
            state = actual; world.Apply(state); world.Animate(state, 0);
        }

        IEnumerator CaptureAgent()
        {
            HasExpedition = true; Screen = CaveScreen.Play; LabVisible = true;
            yield return null;
            yield return new WaitForEndOfFrame();
            CaptureNamed("lab");
            LabVisible = false; HasExpedition = false; Screen = CaveScreen.Title;
            smokeStage = 41;
        }
        IEnumerator CaptureSmoke()
        {
            HasExpedition = false; LabVisible = false;
            foreach (var screen in new[] { CaveScreen.Title, CaveScreen.Caves, CaveScreen.Options, CaveScreen.Play, CaveScreen.Pause })
            {
                Screen = screen;
                HasExpedition = screen == CaveScreen.Play || screen == CaveScreen.Pause;
                yield return new WaitForSecondsRealtime(.15f);
                yield return new WaitForEndOfFrame();
                CaptureNamed(screen.ToString().ToLowerInvariant());
            }
            Screen = CaveScreen.Play;
            yield return new WaitForEndOfFrame();
            CaptureNamed(null);
            smokeStage = 9;
        }
        IEnumerator CaptureResult()
        {
            yield return new WaitForEndOfFrame();
            CaptureNamed("result");
            Back();
            if (Screen != CaveScreen.Title) { FinishSmoke(false, "Results back navigation failed"); yield break; }
            Open(CaveScreen.Caves); Open(CaveScreen.Options); Open(CaveScreen.Lab);
            Back();
            if (Screen != CaveScreen.Options) { FinishSmoke(false, "Lab back navigation failed"); yield break; }
            Back();
            if (Screen != CaveScreen.Caves) { FinishSmoke(false, "Options back navigation failed"); yield break; }
            Back();
            if (Screen != CaveScreen.Title) { FinishSmoke(false, "Caves back navigation failed"); yield break; }
            PlayCave(3);
            smokeStage = 11;
        }
        void CaptureNamed(string suffix)
        {
            if (string.IsNullOrEmpty(capturePath)) return;
            var path = suffix == null ? capturePath : Path.Combine(Path.GetDirectoryName(capturePath), Path.GetFileNameWithoutExtension(capturePath) + "-" + suffix + ".png");
            var texture = ScreenCapture.CaptureScreenshotAsTexture();
            File.WriteAllBytes(path, texture.EncodeToPNG());
            Destroy(texture);
        }
        IEnumerator CaptureReview()
        {
            yield return new WaitForEndOfFrame();
            CaptureNamed(null);
            File.WriteAllText(Path.ChangeExtension(capturePath, ".json"), JsonUtility.ToJson(state, true));
        }
        void FinishSmoke(bool success, string message)
        {
            if (success && referenceSmoke && (!enemyHitChecked || !bonesChecked || !projectilesChecked || !pickupsChecked))
            { success = false; message = "Reference smoke never verified the requested presentation checks"; }
            if (mechanismSmoke && CaveVisualSettings.HudBottom != originalHudBottom) CaveVisualSettings.ToggleHud();
            if (success && mechanismSmoke) message += "; real-input lift boarding and ascent, active green thorn and both HUD/camera positions passed";
            if (success && referenceSmoke) message += "; real-input ceiling-trap trigger, discrete fall, safe dodge and floor impact passed";
            if (success && referenceSmoke) message += "; gravity orientation, airborne poses, contact shadows and normal-state restoration passed (presentation fixtures)";
            if (success && referenceSmoke) message += "; enemy hit poses, anchors and restoration passed (presentation fixtures)";
            if (success && referenceSmoke) message += "; bone poses, signed drift, hard alpha, expiry and restoration passed (presentation fixtures)";
            if (success && referenceSmoke) message += "; capsule poses/facing/anchors passed (presentation fixtures), pickups stayed at authored sites";
            File.WriteAllText(reportPath, "{\"success\":" + (success ? "true" : "false") + ",\"message\":\"" + message + "\"}");
            smoke = false;
            Application.Quit(success ? 0 : 1);
        }
    }
}

using UnityEngine;

namespace CrystalCaves.Pilot
{
    public sealed partial class CaveInterface
    {
        int bindingAction = -1;
        string bindingMessage = "Choose an action to change its keyboard key. ESC always pauses and goes back.";
        float controlsHintUntil = -1;
        static readonly string[] controllerLabels = { "STICK LEFT", "STICK RIGHT", "A / STICK UP", "X", "Y / STICK DOWN", "START" };

        public bool HandleControlsInput()
        {
            if (game.Screen != CaveScreen.Controls) { bindingAction = -1; return false; }
            if (bindingAction >= 0)
            {
                if (Input.GetKeyDown(KeyCode.Escape) || Input.GetKeyDown(KeyCode.JoystickButton1))
                { bindingAction = -1; bindingMessage = "Key change cancelled. Your controls are unchanged."; }
                else if (CaveControls.Capture(out var key))
                {
                    var action = CaveControls.Actions[bindingAction];
                    if (CaveControls.TryBind(action, key, out bindingMessage)) bindingAction = -1;
                }
                return true;
            }
            navigation.Poll(CaveControls.Actions.Length + 3);
            if (navigation.Back) { game.Back(); navigation.Clear(); }
            // Menu navigation is handled here; global keys still work when
            // we are not capturing a binding (for example F12 screenshots).
            return false;
        }
        void Controls()
        {
            Scrim();
            Heading("KEYBOARD / CONTROLLER", "MAKE EVERY MOVE COUNT.", "Collect every crystal. Fire while jumping. Bump cracked diamond blocks from below to reveal hidden crystals.");
            Text(112, 252, 350, 34, "ACTION", 17, dim, true);
            Text(490, 252, 590, 34, "KEYBOARD / SELECT TO CHANGE", 17, gold, true);
            Text(1112, 252, 350, 34, "CONTROLLER", 17, teal, true);
            for (var i = 0; i < CaveControls.Actions.Length; i++)
            {
                var action = CaveControls.Actions[i]; var y = 302 + i * 73;
                Fill(new Rect(112, y, 1376, 62), new Color(0, 0, .12f, .94f));
                Text(132, y, 345, 62, CaveControls.Name(action), 23, paper, true);
                if (Button(490, y + 4, 586, 54, bindingAction == i ? "PRESS ONE KEY… / ESC CANCEL" : CaveControls.Label(action), bindingAction < 0, bindingAction == i))
                { bindingAction = i; bindingMessage = "Press one key for " + CaveControls.Name(action) + ". ESC cancels."; }
                Text(1112, y, 350, 62, controllerLabels[i], 21, teal, true);
            }
            if (Button(112, 790, 360, 61, "RESTORE DEFAULT KEYS", bindingAction < 0))
            { CaveSettings.Change(CaveControls.Reset, false); bindingMessage = "Default keyboard controls restored."; }
            if (Button(498, 790, 360, 61, "BACK", bindingAction < 0, true)) game.Back();
            Text(914, 790, 574, 70, "Hold up/down to climb chains.\nRelease the controls to hold your place.", 21, dim);
            Plate(new Rect(112, 884, 1376, 68));
            Text(132, 891, 1336, 54, bindingMessage, 21, bindingAction >= 0 ? gold : paper);
            Text(112, 961, 1376, 25, navigation.Controller ? "STICK SELECT    A CHANGE    B BACK" : "↑ ↓ SELECT    ENTER CHANGE    ESC BACK", 16, dim);
        }
        void DrawControlsHint()
        {
            if (S.realm == "mine" || S.mode != "human" || game.ScreenshotMode || game.CapturingScreenshot) return;
            if (!CaveSettings.Data.controlsHintSeen)
            {
                controlsHintUntil = Time.unscaledTime + 8;
                CaveSettings.Change(d => d.controlsHintSeen = true, false);
            }
            if (Time.unscaledTime >= controlsHintUntil) return;
            var hint = CaveControls.Hint;
            var x = 12f; var y = CaveViewport.WorldTop + 8;
            Fill(new Rect(x - 4, y - 4, Mathf.Min(CaveViewport.Width - 16, hint.Length * 6 + 8), 17), new Color(0, 0, 0, .92f));
            PixelText(x, y, hint, 1, paper);
        }
    }
}

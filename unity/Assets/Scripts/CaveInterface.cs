using System.Collections.Generic;
using UnityEngine;

namespace CrystalCaves.Pilot
{
    // Menus belong to the game; policy inspection is a deliberately separate view.
    public sealed partial class CaveInterface
    {
        readonly CavePilot game;
        readonly Color paper = Color.white;
        readonly Color gold = new Color(1, 1, 1f / 3);
        readonly Color dim = new Color(2f / 3, 2f / 3, 2f / 3);
        readonly Color ink = Color.black;
        readonly Color teal = new Color(1f / 3, 1, 1);
        readonly Texture2D crystal, caves, gradient, pixelFont, titleScene;
        readonly Texture2D vignette;
        readonly CaveCatalog catalog;
        Texture2D white, heart;
        float healthPulse, ammoPulse, gemPulse, screenSince;
        CaveScreen lastScreen;
        CaveSnapshot previous;
        sealed class Flight { public Vector2 start; public float time; }
        readonly List<Flight> flights = new List<Flight>();
        string CrystalName => "crystal_blue";
        Font regular, bold;
        GUIStyle label, hit, slider, thumb;
        CaveSnapshot S => game.State;

        public CaveInterface(CavePilot owner)
        {
            game = owner;
            pixelFont = Resources.Load<Texture2D>("Interface/pixel_font");
            crystal = Resources.Load<Texture2D>("Interface/crystal");
            caves = Resources.Load<Texture2D>("Interface/caves");
            titleScene = Resources.Load<Texture2D>("Interface/title_scene");
            vignette = Resources.Load<Texture2D>("Environment/vignette");
            var asset = Resources.Load<TextAsset>("CaveCatalog");
            if (asset != null) catalog = JsonUtility.FromJson<CaveCatalog>(asset.text);
            gradient = new Texture2D(128, 1, TextureFormat.RGBA32, false);
            for (var x = 0; x < 128; x++) gradient.SetPixel(x, 0, new Color(ink.r, ink.g, ink.b, Mathf.Lerp(.98f, .15f, Mathf.Pow(x / 127f, 1.7f))));
            gradient.Apply();
        }

        void Initialize()
        {
            if (white != null) return;
            white = new Texture2D(1, 1);
            white.SetPixel(0, 0, Color.white); white.Apply();
            heart = Resources.Load<Texture2D>("Interface/heart");
            regular = Resources.Load<Font>("Fonts/ChakraPetch-Regular");
            bold = Resources.Load<Font>("Fonts/ChakraPetch-SemiBold");
            label = new GUIStyle(GUI.skin.label) { font = regular, padding = new RectOffset(0, 0, 0, 0), wordWrap = true };
            hit = new GUIStyle(GUI.skin.button) { padding = new RectOffset(0, 0, 0, 0) };
            hit.normal.textColor = hit.hover.textColor = hit.active.textColor = gold;
            hit.fontSize = 22;
            hit.normal.background = hit.hover.background = hit.active.background = hit.focused.background = null;
            slider = new GUIStyle(GUI.skin.horizontalSlider) { fixedHeight = 7, margin = new RectOffset(0, 0, 12, 0), padding = new RectOffset(0, 0, 0, 0) };
            slider.normal.background = white;
            thumb = new GUIStyle(GUI.skin.horizontalSliderThumb) { fixedWidth = 18, fixedHeight = 30 };
            thumb.normal.background = thumb.hover.background = thumb.active.background = white;
        }

        void Fill(Rect rect, Color color)
        { GUI.color = color; GUI.DrawTexture(rect, white); GUI.color = Color.white; }
        void Text(float x, float y, float width, float height, string text, int size = 20, Color? color = null, bool heavy = false, TextAnchor align = TextAnchor.MiddleLeft)
        {
            label.font = heavy ? bold : regular;
            label.fontSize = size; label.alignment = align;
            label.normal.textColor = color ?? paper;
            GUI.Label(new Rect(x, y, width, height), text, label);
        }
        void PixelText(float x, float y, string text, int scale = 3, Color? tint = null)
        {
            GUI.color = tint ?? paper;
            foreach (var character in text.ToUpperInvariant())
            {
                var code = Mathf.Clamp(character - 32, 0, 95);
                var uv = new Rect(code % 16 / 16f, 1 - (code / 16 + 1) / 6f, 1f / 16, 1f / 6);
                GUI.DrawTextureWithTexCoords(new Rect(x, y, 6 * scale, 8 * scale), pixelFont, uv);
                x += 6 * scale;
            }
            GUI.color = Color.white;
        }
        void Image(Rect rect, Texture2D image, Color? tint = null, ScaleMode scale = ScaleMode.ScaleToFit)
        {
            if (image == null) return;
            GUI.color = tint ?? Color.white; GUI.DrawTexture(rect, image, scale); GUI.color = Color.white;
        }
        void Rule(float x, float y, float width, Color? color = null) { Fill(new Rect(x, y, width, 1), color ?? new Color(.23f, .29f, .28f)); }
        bool Hover(Rect rect) => rect.Contains(new Vector2(Input.mousePosition.x / UnityEngine.Screen.width * 1600, (1 - Input.mousePosition.y / UnityEngine.Screen.height) * 1000));
        bool Button(float x, float y, float width, float height, string text, bool enabled = true, bool primary = false)
        {
            var rect = new Rect(x, y, width, height);
            var index = menuIndex++;
            var focused = game.Screen != CaveScreen.Options && navigation.Focus == index;
            var hover = enabled && (Hover(rect) || focused);
            Fill(rect, enabled ? primary ? hover ? paper : gold : hover ? new Color(0, .33f, .33f) : new Color(0, 0, .33f, .98f) : new Color(.12f, .15f, .15f));
            Fill(new Rect(x, y + height - 3, width, 3), enabled ? primary ? new Color(2f / 3, 1f / 3, 0) : new Color(1f / 3, 1f / 3, 1) : ink);
            Text(x + 22, y, width - 44, height - 3, text, height >= 65 ? 25 : 19, enabled ? primary ? ink : paper : dim, true);
            GUI.enabled = enabled;
            var clicked = GUI.Button(rect, GUIContent.none, hit) || enabled && game.Screen != CaveScreen.Options && navigation.Activate(index);
            GUI.enabled = true;
            if (focused && enabled) Fill(new Rect(x, y, 4, height), gold);
            return clicked;
        }
        bool Link(float x, float y, float width, string text, Color? color = null)
        {
            var rect = new Rect(x, y, width, 36);
            Text(x, y, width, 36, text, 17, Hover(rect) ? gold : color ?? dim, true);
            var index = menuIndex++;
            if (game.Screen != CaveScreen.Options && navigation.Focus == index) Fill(new Rect(x - 8, y + 10, 3, 16), gold);
            return GUI.Button(rect, GUIContent.none, hit) || game.Screen != CaveScreen.Options && navigation.Activate(index);
        }
        void Plate(Rect rect, Color? accent = null)
        {
            Fill(rect, new Color(ink.r, ink.g, ink.b, .88f));
            Fill(new Rect(rect.x, rect.y, rect.width, 1), accent ?? new Color(.4f, .66f, .7f, .45f));
            Fill(new Rect(rect.x, rect.y + rect.height - 1, rect.width, 1), new Color(.12f, .28f, .33f, .6f));
        }
        float Pulse(float since) => Mathf.Max(0, 1 - (Time.unscaledTime - since) / .65f);
        public void OnSnapshot(CaveSnapshot next)
        {
            if (previous != null && previous.episode == next.episode)
            {
                if (next.health < previous.health) healthPulse = Time.unscaledTime;
                if (next.ammo != previous.ammo) ammoPulse = Time.unscaledTime;
                if (next.crystals < previous.crystals)
                {
                    gemPulse = Time.unscaledTime;
                    var ids = new HashSet<string>(); foreach (var entity in next.entities) ids.Add(entity.id);
                    foreach (var entity in previous.entities)
                    {
                        if (!entity.id.StartsWith("crystal_") || ids.Contains(entity.id)) continue;
                        var point = game.GameCamera.WorldToScreenPoint(CaveWorld.Position(entity.x + 16, entity.y + 16));
                        flights.Add(new Flight { start = CaveViewport.Point(point), time = Time.unscaledTime });
                    }
                }
            }
            else { flights.Clear(); healthPulse = ammoPulse = gemPulse = -10; }
            previous = next;
        }
        void Scrim(float alpha = .94f) { Fill(new Rect(0, 0, 1600, 1000), new Color(ink.r, ink.g, ink.b, alpha)); }
        void Heading(string eyebrow, string title, string description)
        {
            Text(112, 84, 1150, 25, eyebrow, 16, gold, true);
            PixelText(112, 140, title, 7);
            Text(112, 205, 1300, 29, description, 20, dim);
            if (Link(1390, 83, 115, "ESC  BACK")) game.Back();
        }

        public void Draw()
        {
            Initialize();
            menuIndex = 0;
            if (lastScreen != game.Screen) { navigation.Reset(); settingsLoaded = false; }
            GUI.matrix = CaveViewport.Matrix;
            Fill(new Rect(0, CaveViewport.HudY, CaveViewport.Width, CaveViewport.HudHeight), Color.black);
            GUI.matrix = Matrix4x4.Scale(new Vector3(UnityEngine.Screen.width / 1600f, UnityEngine.Screen.height / 1000f, 1));
            if (S == null) return;
            if (lastScreen != game.Screen) { lastScreen = game.Screen; screenSince = Time.unscaledTime; }
            Image(new Rect(0, 0, 1600, 1000), vignette, null, ScaleMode.StretchToFill);
            switch (game.Screen)
            {
                case CaveScreen.Title: Title(); break;
                case CaveScreen.Play:
                    GUI.matrix = Matrix4x4.identity;
                    var frame = CaveViewport.Frame;
                    Fill(new Rect(0, 0, UnityEngine.Screen.width, frame.y), Color.black);
                    Fill(new Rect(0, frame.yMax, UnityEngine.Screen.width, UnityEngine.Screen.height - frame.yMax), Color.black);
                    Fill(new Rect(0, frame.y, frame.x, frame.height), Color.black);
                    Fill(new Rect(frame.xMax, frame.y, UnityEngine.Screen.width - frame.xMax, frame.height), Color.black);
                    GUI.matrix = CaveViewport.Matrix;
                    if (!game.HideHud) Hud();
                    GUI.matrix = Matrix4x4.Scale(new Vector3(UnityEngine.Screen.width / 1600f, UnityEngine.Screen.height / 1000f, 1));
                    break;
                case CaveScreen.Pause: Pause(); break;
                case CaveScreen.Caves: CaveSelect(); break;
                case CaveScreen.Options: Options(); break;
                case CaveScreen.Controls: Controls(); break;
                case CaveScreen.Result: Result(); break;
                case CaveScreen.Lab: Scrim(); Lab(false); break;
            }
            if (game.LabVisible && game.Screen == CaveScreen.Play && !game.CapturingScreenshot) Lab(true);
            DrawPresentationOverlay();
            if (Event.current.type == EventType.Repaint) navigation.Clear();
        }

        void Title()
        {
            Image(new Rect(0, 0, 1600, 1000), gradient, null, ScaleMode.StretchToFill);
            Fill(new Rect(0, 0, 740, 1000), new Color(0, 0, 0, .91f));
            PixelText(112, 128, "A CRYSTAL COLLECTING ADVENTURE", 3, teal);
            Image(new Rect(108, 226, crystal.width * 4, crystal.height * 4), crystal);
            Image(new Rect(108, 358, caves.width * 4, caves.height * 4), caves, gold);
            Text(112, 501, 620, 64, "Sixteen caves. Every crystal. One way out.\nGrab your raygun and get exploring.", 24, paper);
            var ready = game.Connected && !game.OpeningCave;
            var playLabel = game.OpeningCave && game.Connected ? "OPENING YOUR CAVE…" : game.HasExpedition && !S.done ? "CONTINUE EXPLORING   →" : "EXPLORE THE MAIN MINE   →";
            if (Button(112, 608, 450, 76, playLabel, ready, true)) game.Play();
            if (Button(112, 708, 214, 61, "CAVES   [C]")) game.Open(CaveScreen.Caves);
            if (Button(348, 708, 214, 61, "OPTIONS   [O]")) game.Open(CaveScreen.Options);
            if (Link(112, 778, 450, "CONTROLS / KEYBOARD + CONTROLLER")) game.Open(CaveScreen.Controls);
            var cleared = 0; for (var i = 0; i < 16; i++) cleared += PlayerPrefs.GetInt("cave-cleared-" + i, 0);
            Rule(112, 815, 450, dim);
            PixelText(112, 839, cleared + " / 16 CAVES CLEARED", 2, dim);
            if (!game.Connected) Text(112, 878, 610, 36, "Preparing your cave. Start with the game launcher.", 17, dim);
            else if (!string.IsNullOrEmpty(game.Warning)) Text(112, 878, 610, 36, game.Warning, 17, gold);
            if (Link(112, 924, 215, "AI LAB   /   F2")) game.Open(CaveScreen.Lab);
            if (Link(348, 924, 120, "QUIT")) Application.Quit();
            Image(new Rect(840, 250, 640, 448), titleScene);
            Fill(new Rect(925, 784, 565, 92), new Color(0, 0, 0, .94f));
            PixelText(951, 809, "THE MINES ARE WAITING.", 3, gold);
            Text(951, 844, 520, 24, "A bright arcade world, one cave at a time.", 18, paper);
        }

        void Heart(float x, float y, bool alive)
        {
            Image(new Rect(x, y, 36, 32), heart, alive ? Color.white : new Color(.25f, .25f, .25f));
        }
        void Hud()
        {
            var y = CaveViewport.HudY;
            Fill(new Rect(0, y, CaveViewport.Width, CaveViewport.HudHeight), Color.black);
            if (CaveSettings.Data.hudScale > 1 || CaveSettings.Data.safeMargin > 0 || CaveSettings.Data.highContrast) ReadableHud(y);
            else if (S.realm == "mine")
            {
                PixelText(12, y + 10, "$ " + S.score.ToString("D6"), 2, new Color(1f / 3, 1, 1f / 3));
                PixelText(165, y + 10, S.cleared_caves + "/16", 2, gold);
                PixelText(236, y + 14, "CLEARED", 1, dim);
                Image(new Rect(287, y + 6, 26, 22), Resources.Load<Texture2D>("Interface/raygun"));
                PixelText(323, y + 10, S.ammo.ToString(), 2, new Color(1f / 3, 1, 1f / 3));
                for (var i = 0; i < 3; i++) Image(new Rect(391 + i * 21, y + 8, 18, 16), heart, Color.white);
                PixelText(481, y + 12, "MAIN MINE", 2, dim);
                if (S.near_entrance >= 0)
                {
                    Fill(new Rect(150, CaveViewport.HintY, 340, 23), new Color(0, 0, 0, .94f));
                    PixelText(160, CaveViewport.HintY + 5, "CAVE " + (S.near_entrance + 1).ToString("00") + " / " + S.levels[S.near_entrance] + "   [E / ENTER]", 1, gold);
                }
                else PixelText(180, CaveViewport.WorldTop + 10, "FIND A DOOR / UP-DOWN TO CLIMB CHAINS", 1, dim);
                foreach (var entrance in S.entities)
                {
                    if (!entrance.id.StartsWith("entrance_")) continue;
                    var point = CaveViewport.Point(game.GameCamera.WorldToScreenPoint(CaveWorld.Position(entrance.x + 7, entrance.y - 10)));
                    if (point.x > 0 && point.x < CaveViewport.Width - 16 && point.y > CaveViewport.WorldTop + 3 && point.y < CaveViewport.WorldBottom - 8)
                        PixelText(point.x, point.y, (int.Parse(entrance.id.Substring(9)) + 1).ToString("00"), 1, gold);
                }
            }
            else
            {
                PixelText(12, y + 10, "$ " + S.score.ToString("D6"), 2, new Color(1f / 3, 1, 1f / 3));
                Image(new Rect(165, y + 6, 22, 22), Resources.Load<Texture2D>("Sprites/" + CrystalName));
                PixelText(195, y + 10, S.crystals + "/" + S.initial_crystals, 2, gold);
                Image(new Rect(287, y + 6, 26, 22), Resources.Load<Texture2D>("Interface/raygun"));
                PixelText(323, y + 10, S.ammo.ToString(), 2, new Color(1f / 3, 1, 1f / 3));
                for (var i = 0; i < 3; i++) Image(new Rect(391 + i * 21, y + 8, 18, 16), heart, S.health > i ? Color.white : new Color(.25f, .25f, .25f));
                PixelText(481, y + 12, "CAVE " + (S.level + 1).ToString("00"), 2, dim);
                if (S.mode == "ai") PixelText(12, CaveViewport.WorldTop + 12, "AI AT THE CONTROLS / F2", 1, teal);
            }
            PixelText(CaveViewport.Width - 25 - CaveSettings.Data.safeMargin, y + 11, "II", 2, Hover(new Rect(CaveViewport.Width - 34 - CaveSettings.Data.safeMargin, y + 2, 32, 28)) ? gold : paper);
            if (GUI.Button(new Rect(CaveViewport.Width - 34 - CaveSettings.Data.safeMargin, y + 2, 32, 28), GUIContent.none, hit)) game.Back();
            var allCrystals = S.realm != "mine" && S.initial_crystals > 0 && S.crystals == 0;
            if (allCrystals) PixelText(255, CaveViewport.HintY + 12, "ALL CRYSTALS / EXIT OPEN", 1, new Color(1f / 3, 1, 1f / 3));
            for (var i = flights.Count - 1; i >= 0; i--)
            {
                var progress = (Time.unscaledTime - flights[i].time) / .65f;
                if (progress >= 1) { flights.RemoveAt(i); continue; }
                if (!CaveVisualSettings.Motion || !CaveSettings.Data.particles || CaveSettings.Data.effectDensity == 0) continue;
                var point = Vector2.Lerp(flights[i].start, new Vector2(175, y + 16), progress * progress);
                point.y -= Mathf.Sin(progress * Mathf.PI) * 24;
                Image(new Rect(Mathf.Round(point.x) - 8, Mathf.Round(point.y) - 8, 16, 16), Resources.Load<Texture2D>("Sprites/" + CrystalName));
            }
            if (S.effects != null) foreach (var effect in S.effects)
            {
                if (string.IsNullOrEmpty(effect.text)) continue;
                // Keep feedback labels above the authored action/pulse pixels.
                var lift = effect.text == "OUCH" ? 24 : effect.kind == "bones" || effect.kind == "slime_pulse" ? 12 : 0;
                var textY = effect.y - (effect.max_ttl - effect.ttl) * .7f - lift;
                var point = CaveViewport.Point(game.GameCamera.WorldToScreenPoint(CaveWorld.Position(effect.x, textY)));
                if (point.x > 12 && point.x < CaveViewport.Width - 40 && point.y > CaveViewport.WorldTop + 12 && point.y < CaveViewport.WorldBottom - 12)
                {
                    var text = CavePowerFeedback.PickupLabel(effect.kind, effect.text);
                    var textX = text == effect.text ? point.x - 12
                        : Mathf.Clamp(point.x - text.Length * 3, 12, Mathf.Max(12, CaveViewport.Width - text.Length * 6 - 12));
                    PixelText(textX + 1, point.y - 8 + 1, text, 1, ink);
                    PixelText(textX, point.y - 8, text, 1, gold);
                }
            }
            if (CaveSettings.Data.itemMarkers) { DrawItemMarkers(); DrawSecretCacheMarkers(); }
            DrawControlsHint();
            DrawPowerStatus();
            if (S.realm != "mine") CaveBorder(allCrystals);
        }

        void CaveBorder(bool ready)
        {
            var color = ready ? new Color(1f / 3, 1, 1f / 3) : new Color(1, 1f / 3, 1f / 3);
            var width = CaveViewport.Width; var height = CaveViewport.Height;
            Fill(new Rect(0, 0, width, 2), color);
            Fill(new Rect(0, height - 2, width, 2), color);
            Fill(new Rect(0, 0, 2, height), color);
            Fill(new Rect(width - 2, 0, 2, height), color);
        }

        void Pause()
        {
            Scrim(.9f);
            Heading("EXPEDITION PAUSED", "TAKE A BREATHER.", S.realm == "mine" ? "MAIN MINE" : (S.level + 1).ToString("00") + " / " + S.level_name);
            if (Button(112, 306, 450, 76, "BACK TO EXPLORING   →", game.Connected && !S.done, true)) game.Resume();
            if (Button(112, 406, 450, 61, "SETTINGS   [O]")) game.Open(CaveScreen.Options);
            if (Button(112, 491, 450, 61, S.realm == "mine" ? "CHOOSE A CAVE   [C]" : "RETURN TO THE MAIN MINE", game.Connected && !game.OpeningCave))
            { if (S.realm == "mine") game.Open(CaveScreen.Caves); else game.OpenMine(); }
            if (Button(112, 576, 450, 61, "CONTROLS")) game.Open(CaveScreen.Controls);
            if (Link(112, 679, 270, "RETURN TO MAIN MENU")) game.Home();
            Rule(790, 322, 630);
            Text(790, 350, 590, 35, "YOUR EXPEDITION", 17, gold, true);
            Text(790, 414, 590, 60, S.realm == "mine" ? S.cleared_caves + " / 16 CAVES CLEARED" : (S.initial_crystals - S.crystals) + " / " + S.initial_crystals + " CRYSTALS", 36, paper, true);
            Text(790, 494, 590, 40, S.score.ToString("N0") + " POINTS", 23, dim);
            Text(790, 578, 590, 100, S.realm == "mine" ? "Walk to a door and press E or Enter.\nUp/Down climb chains; release to hold." : "Collect every crystal to unlock the exit.\nWatch your footing. Make every shot count.", 23, dim);
            if (!string.IsNullOrEmpty(game.Warning)) Text(112, 760, 900, 70, game.Warning, 22, gold);
            if (Link(112, 924, 240, "AI LAB   /   F2")) game.Open(CaveScreen.Lab);
        }

        void CaveSelect()
        {
            Scrim();
            Heading("THE EXPEDITION / 16 CAVES", "PICK YOUR DESCENT.", "Choose a cave. Collect its crystals. Find your way out.");
            for (var i = 0; i < 16; i++)
            {
                var x = 112 + i % 4 * 348;
                var y = 263 + i / 4 * 150;
                var rect = new Rect(x, y, 322, 130);
                var selected = game.SelectedCave == i;
                Fill(rect, selected ? gold : new Color(.16f, .22f, .22f));
                var atlas = Resources.Load<Texture2D>("Interface/cave_" + i);
                Image(new Rect(x + 3, y + 3, 316, 124), atlas, new Color(.95f, .95f, .95f), ScaleMode.ScaleAndCrop);
                Fill(new Rect(x + 3, y + 74, 316, 53), new Color(0, 0, 0, .93f));
                Fill(new Rect(x + 10, y + 9, 130, 28), new Color(0, 0, 0, .86f));
                Text(x + 18, y + 13, 255, 24, (i + 1).ToString("00") + (PlayerPrefs.GetInt("cave-cleared-" + i, 0) == 1 ? "   /   CLEARED" : ""), 15, selected ? gold : dim, true);
                var name = S.levels != null && i < S.levels.Length ? S.levels[i] : "Cave " + (i + 1);
                PixelText(x + 18, y + 80, name.ToUpperInvariant(), 2, paper);
                var count = catalog != null && i < catalog.caves.Length ? catalog.caves[i].crystals : 0;
                Text(x + 18, y + 102, 280, 22, count + " CRYSTALS" + (selected ? "   /   SELECTED" : ""), 14, selected ? gold : dim);
                if (GUI.Button(rect, GUIContent.none, hit)) game.Select(i);
            }
            if (Button(112, 891, 436, 65, "EXPLORE CAVE " + (game.SelectedCave + 1).ToString("00") + "   →", game.Connected && !game.OpeningCave, true)) game.PlayCave(game.SelectedCave);
            Text(598, 903, 720, 40, "ARROW KEYS  CHOOSE     /     ENTER  EXPLORE", 17, dim);
        }

        void Result()
        {
            Scrim(.88f);
            var title = S.won ? "THE CAVE IS YOURS." : S.end_reason == "timeout" ? "OUT OF TIME." : S.end_reason == "stalled" ? "KEEP ON EXPLORING." : "ONE MORE TRY?";
            var detail = S.won ? "Every crystal collected. Another story from the depths." : "A fresh start. Another route. You know the cave a little better.";
            Heading(S.won ? "EXPEDITION COMPLETE" : "EXPEDITION ENDED", title, detail);
            var elapsed = Time.unscaledTime - screenSince;
            var reveal = CaveVisualSettings.Motion ? Mathf.SmoothStep(0, 1, elapsed / .7f) : 1;
            Text(112, 318, 650, 65, Mathf.RoundToInt((S.initial_crystals - S.crystals) * reveal) + " / " + S.initial_crystals + " CRYSTALS", 38, gold, true);
            Text(112, 418, 650, 50, Mathf.RoundToInt(S.score * reveal).ToString("N0") + " POINTS", 29, paper, true);
            Text(112, 494, 800, 50, (S.level + 1).ToString("00") + " / " + S.level_name + (S.won && !S.human_only ? "   ·   AI ASSISTED" : ""), 23, dim);
            Image(new Rect(1040, 334, 192, 256), Resources.Load<Texture2D>("Sprites/mylo_" + (S.won ? "idle" : "hurt")));
            if (S.won)
            {
                Image(new Rect(1250, 264, 100, 145), Resources.Load<Texture2D>("Sprites/" + CrystalName));
                Text(926, 716, 530, 48, "CAVE " + (S.level + 1).ToString("00") + "  /  COMPLETE", 25, gold, true, TextAnchor.MiddleCenter);
            }
            else Text(926, 716, 530, 48, "THE NEXT ROUTE IS WAITING.", 20, dim, true, TextAnchor.MiddleCenter);
            if (Button(112, 611, 450, 76, S.won ? "RETURN TO THE MAIN MINE   →" : "TRY THIS CAVE AGAIN   →", game.Connected && !game.OpeningCave, true)) { if (S.won) game.OpenMine(); else game.PlayCave(S.level); }
            if (Button(112, 711, 450, 61, "CHOOSE A CAVE")) game.Open(CaveScreen.Caves);
            if (Link(112, 837, 270, "RETURN TO MAIN MENU")) game.Home();
        }

        void Lab(bool overlay)
        {
            var x = overlay ? 1018 : 880;
            var width = overlay ? 550 : 650;
            if (!overlay)
            {
                Text(112, 128, 630, 30, "THE OPTIONAL AI LAB", 17, teal, true);
                Text(112, 213, 680, 170, "PLAY FIRST.\nEXPERIMENT LATER.", 57, paper, true);
                Text(112, 435, 620, 156, "Watch a trained agent explore the same caves. Take the controls whenever you want. Your game is always here.", 26, dim);
                if (Button(112, 658, 450, 65, "BACK TO THE GAME   →", true, true)) game.Back();
            }
            Fill(new Rect(x, 112, width, 837), new Color(.035f, .065f, .075f, .98f));
            Fill(new Rect(x, 112, 3, 837), teal);
            Text(x + 30, 138, width - 120, 29, "AI LAB", 23, teal, true);
            if (Link(x + width - 90, 134, 65, "CLOSE")) game.CloseLab();
            Text(x + 30, 188, width - 60, 30, S.mode == "ai" ? "AGENT AT THE CONTROLS" : "YOU ARE AT THE CONTROLS", 17, paper, true);
            Rule(x + 30, 240, width - 60);
            if (S.ai_available)
            {
                Text(x + 30, 258, width - 60, 68, "CHECKPOINT\n" + S.policy_name, 17, dim);
                if (Button(x + 30, 346, width - 60, 61, S.mode == "ai" ? "TAKE THE CONTROLS" : "WATCH THE AGENT", game.Connected && !game.OpeningCave && !S.done, S.mode != "ai"))
                { if (S.mode == "ai") game.TakeControl(); else game.WatchAgent(); }
                Text(x + 30, 432, width - 60, 27, "ACTION VALUES", 15, teal, true);
                if (S.q_values != null && S.q_values.Length > 0)
                {
                    float lo = S.q_values[0], hi = lo;
                    foreach (var value in S.q_values) { lo = Mathf.Min(lo, value); hi = Mathf.Max(hi, value); }
                    for (var i = 0; i < S.q_values.Length; i++)
                    {
                        var y = 473 + i * 37;
                        var name = S.action_labels != null && i < S.action_labels.Length ? S.action_labels[i] : "Action " + i;
                        Text(x + 30, y, 168, 29, name, 14, S.action == i ? paper : dim, S.action == i);
                        Fill(new Rect(x + 208, y + 10, width - 313, 9), new Color(.14f, .2f, .21f));
                        Fill(new Rect(x + 208, y + 10, (width - 313) * Mathf.Max(.02f, (S.q_values[i] - lo) / Mathf.Max(.001f, hi - lo)), 9), S.action == i ? teal : new Color(.27f, .4f, .42f));
                        Text(x + width - 99, y, 69, 29, S.q_values[i].ToString("F2"), 14, dim, false, TextAnchor.MiddleRight);
                    }
                }
                else Text(x + 30, 491, width - 60, 85, "Watch the agent to see how it scores each possible move.", 21, dim);
                Text(x + 30, 875, width - 60, 43, S.done ? "Restart a cave to watch another attempt." : "ENTER  WATCH / TAKE CONTROL   ·   F2  CLOSE", 14, dim);
            }
            else
            {
                Text(x + 30, 279, width - 60, 62, "NO AGENT LOADED", 28, paper, true);
                Text(x + 30, 382, width - 60, 153, "You can explore every cave and save your progress without an AI agent.", 25, dim);
                Text(x + 30, 578, width - 60, 131, "To experiment with AI, enable a compatible checkpoint when launching the game. The Unity README has the details.", 21, dim);
                if (Button(x + 30, 799, width - 60, 65, "BACK TO EXPLORING   →", true, true)) game.CloseLab();
            }
        }
    }
}

using System;
using System.Collections.Generic;
using UnityEngine;

namespace CrystalCaves.Pilot
{
    public sealed partial class CaveInterface
    {
        readonly CaveMenuNavigation navigation = new CaveMenuNavigation();
        readonly List<Option> options = new List<Option>();
        sealed class Option
        {
            public string name, value, help;
            public Action<int> edit;
            public bool enabled = true;
        }
        int settingsTab, menuIndex;
        bool settingsLoaded, stagedFullscreen;
        int stagedWidth, stagedHeight;
        Vector2 lastMouse;
        public int SettingsTab => settingsTab;
        public void SetSettingsTab(int tab) { settingsTab = ((tab % 4) + 4) % 4; navigation.Reset(); }
        public void Navigate(int move, int adjust, bool submit, bool back, int tab)
        {
            BuildOptions(); navigation.Apply(options.Count + 4, move, adjust, submit, back, tab);
            DispatchSettingsNavigation();
        }
        public void HandleMenuInput()
        {
            if (game.Presentation.Pending)
            {
                if (Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.JoystickButton0)) game.Presentation.Keep();
                else if (Input.GetKeyDown(KeyCode.Escape) || Input.GetKeyDown(KeyCode.JoystickButton1)) game.Presentation.Revert();
                return;
            }
            if (game.Screen == CaveScreen.Options)
            {
                LoadStagedDisplay(); BuildOptions(); navigation.Poll(options.Count + 4); DispatchSettingsNavigation();
            }
            else if (game.Screen == CaveScreen.Caves)
            {
                navigation.PointAt(0); navigation.Poll(16);
                if (navigation.Focus!=0) game.Select(game.SelectedCave+(navigation.Focus==15?-4:4));
                if (navigation.Adjust!=0) game.Select(game.SelectedCave+navigation.Adjustment(navigation.Focus));
                if (navigation.Activate(navigation.Focus)) game.PlayCave(game.SelectedCave);
                if (navigation.Back) game.Back(); navigation.Clear();
            }
            else
            {
                var count = game.Screen == CaveScreen.Title ? 6 : game.Screen == CaveScreen.Pause ? 7 : game.Screen == CaveScreen.Result ? 4 : menuIndex;
                navigation.Poll(count);
                if (navigation.Back) game.Back();
            }
        }
        void DispatchSettingsNavigation()
        {
            if (navigation.Back) { game.Back(); navigation.Clear(); return; }
            if (navigation.TabDelta != 0) { SetSettingsTab(settingsTab + navigation.TabDelta); navigation.Clear(); return; }
            var index = navigation.Focus;
            var amount = navigation.Adjustment(index);
            if (navigation.Activate(index)) amount = 1;
            if (amount == 0) return;
            if (index < options.Count) { if (options[index].enabled) options[index].edit(amount); }
            else FooterAction(index - options.Count);
        }
        void LoadStagedDisplay()
        {
            if (settingsLoaded) return;
            stagedWidth = CaveSettings.Data.width; stagedHeight = CaveSettings.Data.height; stagedFullscreen = CaveSettings.Data.fullscreen;
            settingsLoaded = true;
        }
        void FooterAction(int index)
        {
            if (index == 0) { CaveSettings.Reset(); stagedWidth=1600; stagedHeight=1000; stagedFullscreen=false; flights.Clear(); }
            if (index == 1) game.StartScreenshotMode();
            if (index == 2 && game.Connected && game.HasExpedition && S.realm != "mine" && !game.OpeningCave) game.PlayCave(S.level);
            if (index == 3) game.Back();
        }
        void Add(string name, string value, string help, Action<int> edit, bool enabled = true)
        { options.Add(new Option { name = name, value = value, help = help, edit = edit, enabled = enabled }); }
        void Toggle(string name, bool value, string help, Action<CaveSettingsData> edit, bool custom = true)
        { Add(name, value ? "ON" : "OFF", help, amount => CaveSettings.Change(edit, custom)); }
        void Number(string name, int value, int min, int max, int step, string help, Action<CaveSettingsData, int> edit, bool custom = true)
        { Add(name, value + "%", help, amount => CaveSettings.Change(d => edit(d, Mathf.Clamp(value + amount * step, min, max)), custom)); }
        static int Cycle(int value, int amount, int count) => ((value + amount) % count + count) % count;
        void BuildOptions()
        {
            options.Clear(); var d = CaveSettings.Data;
            if (settingsTab == 0)
            {
                Add("WINDOW MODE", stagedFullscreen ? "FULLSCREEN" : "WINDOWED", "Fullscreen uses your display's native mode. Window size is remembered separately. Apply, then keep within 15 seconds.", n => stagedFullscreen = !stagedFullscreen);
                var sizes = new[] { new Vector2Int(640,400), new Vector2Int(960,600), new Vector2Int(1280,800), new Vector2Int(1600,1000), new Vector2Int(1920,1080), new Vector2Int(2560,1440) };
                Add("WINDOW SIZE", stagedWidth + " × " + stagedHeight, "Windowed resolution. You can also resize the window by dragging its edges.", n => {
                    var index = Array.FindIndex(sizes, s => s.x == stagedWidth && s.y == stagedHeight);
                    var size = sizes[Cycle(index < 0 ? 2 : index, n, sizes.Length)]; stagedWidth = size.x; stagedHeight = size.y;
                });
                Add("APPLY DISPLAY", "APPLY / CONFIRM", "An unconfirmed change automatically reverts. Escape also restores the previous display.", n => game.Presentation.ApplyDisplay(stagedWidth, stagedHeight, stagedFullscreen));
                Toggle("V-SYNC", d.vSync, "Matches display refresh to prevent tearing. Frame cap takes effect when V-sync is off.", v => v.vSync = !v.vSync);
                var caps = new[] { 0,30,60,90,120,144,165,240,360 };
                Add("FRAME CAP", d.vSync ? "V-SYNC ACTIVE" : d.frameCap == 0 ? "UNLIMITED" : d.frameCap + " FPS", "Presentation speed only. Game movement always runs at 60 simulation steps per second.", n => CaveSettings.Change(v => v.frameCap = caps[Cycle(Array.IndexOf(caps, d.frameCap), n, caps.Length)]), !d.vSync);
                Toggle("FPS COUNTER", d.showFps, "A small counter reports rendered frames per second; it does not change gameplay.", v => v.showFps = !v.showFps, false);
                Add("PIXEL SCALE", d.pixelScale == 0 ? "AUTO INTEGER" : d.pixelScale + "× (" + CaveViewport.Scale + "× FIT)", "Whole pixels only. An oversized choice falls back to the largest scale that fits your window.", n => CaveSettings.Change(v => v.pixelScale = Cycle(d.pixelScale, n, 7), false));
                var aspects = new[] { "ADAPTIVE CLASSIC", "CLASSIC 16:10", "WIDE / LETTERBOX" };
                Add("CAVE FRAMING", aspects[d.aspect], "Adaptive preserves the existing framing. Classic fixes a 640×400 frame. Wide uses available width without stretching sprites.", n => CaveSettings.Change(v => v.aspect = Cycle(d.aspect, n, 3), false));
            }
            else if (settingsTab == 1)
            {
                var presets = new[] { "Current Retro", "Enhanced Retro", "Low Power" };
                Add("PRESET", d.preset.ToUpperInvariant(), "Current Retro keeps today's look. Enhanced adds sparse environmental motion and soft local light. Low Power reduces decorative work.", n => CaveSettings.Change(v => v.ApplyPreset(Cycle(Array.IndexOf(presets, d.preset), n, 3)), false));
                Number("CAMERA SHAKE", d.shake, 0,100,10, "Damage shake strength. Reduced motion overrides it without changing this saved value.", (v,n) => v.shake=n);
                Toggle("PARTICLES", d.particles, "Decorative pickup and shot particles. Projectiles and danger cues remain visible.", v => v.particles=!v.particles);
                Add("EFFECT DENSITY", new[] { "OFF", "LOW", "NORMAL" }[d.effectDensity], "Controls decorative dust, sparks and vapor. Enemies, thorns, acid and bullets stay visible at every setting.", n => CaveSettings.Change(v => v.effectDensity=Cycle(d.effectDensity,n,3)));
                Toggle("GEM GLINTS", d.gemGlints, "The original gem colors and shapes stay intact when glints are disabled.", v => v.gemGlints=!v.gemGlints);
                Toggle("ENVIRONMENT MOTION", d.environment, "Sparse ceiling drips, service lamps, dust and acid vapor. Reduced motion suppresses this layer.", v => v.environment=!v.environment);
                Number("LOCAL LIGHT", d.lighting, 0,50,5, "Restrained torch and crystal light. Zero preserves the evenly lit retro artwork.", (v,n) => v.lighting=n);
                Toggle("CONTACT SHADOWS", d.contactShadows, "Small ground shadows help anchor creatures. They follow the actual supporting surface.", v => v.contactShadows=!v.contactShadows);
                Number("SHADOW STRENGTH", d.shadowStrength, 0,80,5, "Changes ground shadows without affecting collisions or actor silhouettes.", (v,n) => v.shadowStrength=n);
                Number("BRIGHTNESS", d.brightness, 70,130,5, "Adjust the cave until the dark calibration blocks are distinct. HUD and menus keep their original colors.", (v,n) => v.brightness=n);
                Add("CAVE PALETTE", new[] { "ORIGINAL", "WARM", "COOL" }[d.palette], "A subtle cave-only treatment. Crystal identity and the ungraded HUD remain readable.", n => CaveSettings.Change(v => v.palette=Cycle(d.palette,n,3)));
                Number("CRT SCANLINES", d.scanlines, 0,60,5, "Optional native-row scanlines. Off by default. The HUD stays clear.", (v,n) => v.scanlines=n);
                Number("CRT PHOSPHOR", d.phosphor, 0,40,5, "Optional restrained RGB phosphor pattern. No blur or curved screen distortion.", (v,n) => v.phosphor=n);
            }
            else if (settingsTab == 2)
            {
                Number("MASTER VOLUME", d.masterVolume, 0,100,5, "Overall PC-speaker volume. The reference audio and its single-speaker priorities are preserved.", (v,n) => v.masterVolume=n, false);
                Number("EFFECTS VOLUME", d.effectsVolume, 0,100,5, "Adjusts the classic event sounds within master volume.", (v,n) => v.effectsVolume=n, false);
                Toggle("MUTE", d.muted, "Silences audio without losing your volume levels. M also toggles mute.", v => v.muted=!v.muted, false);
                Add("TEST RAYGUN SOUND", "PLAY", "Preview the actual in-game raygun cue at the selected volume.", n => game.Sound.Preview("shoot"));
            }
            else
            {
                Toggle("REDUCED MOTION", d.reducedMotion, "Disables camera shake, decorative motion, particles and glints. Running, lifts, bullets and hazard cues still animate.", v => v.reducedMotion=!v.reducedMotion, false);
                Toggle("DAMAGE FLASHES", d.damageFlashes, "Disable repeated white damage flashes while keeping a steady hurt pose and invulnerability behavior.", v => v.damageFlashes=!v.damageFlashes);
                Add("HUD POSITION", d.hudBottom ? "BOTTOM / CLASSIC" : "TOP / MODERN", "Money, gems, raygun and hearts move together. The camera reserves the matching strip.", n => { CaveVisualSettings.ToggleHud(); flights.Clear(); });
                Add("HUD SIZE", d.hudScale == 1 ? "STANDARD" : "LARGE", "Large uses larger pixel lettering and icons with a taller HUD strip.", n => CaveSettings.Change(v => v.hudScale=d.hudScale==1?2:1, false));
                Add("SAFE MARGIN", d.safeMargin + " PX", "Moves HUD information away from display edges, keeping pixel lettering sharp.", n => CaveSettings.Change(v => v.safeMargin=Mathf.Clamp(d.safeMargin+n*4,0,24), false));
                Toggle("HIGH CONTRAST HUD", d.highContrast, "White numbers and a solid black strip improve readability.", v => v.highContrast=!v.highContrast, false);
                Toggle("ITEM SYMBOLS", d.itemMarkers, "Adds small B/G/Y/R letters to gems and + to health pickups so color is not the only cue.", v => v.itemMarkers=!v.itemMarkers, false);
                Toggle("SCREENSHOT HUD", !d.screenshotHideHud, "Choose whether screenshot mode includes the HUD. F12 saves a normal game screenshot at any time.", v => v.screenshotHideHud=!v.screenshotHideHud, false);
            }
        }
        void Options()
        {
            LoadStagedDisplay(); BuildOptions();

            Plate(new Rect(36,35,858,930));
            PixelText(64,65,"YOUR EXPEDITION. YOUR WAY.",4,gold);
            Text(64,110,810,34,"Settings are saved automatically. Display changes need confirmation.",18,dim);
            var tabs = new[] { "DISPLAY", "GRAPHICS", "SOUND", "ACCESSIBILITY" };
            for (var i=0;i<4;i++)
            {
                var rect = new Rect(64+i*201,164,190,48);
                Fill(rect,i==settingsTab?gold:new Color(0,0,.33f));
                Text(rect.x+12,rect.y,rect.width-24,rect.height,tabs[i],17,i==settingsTab?ink:paper,true);
                if (GUI.Button(rect,GUIContent.none,hit)) SetSettingsTab(i);
            }
            var rowHeight = options.Count > 10 ? 44 : 62;
            for (var i=0;i<options.Count;i++)
            {
                var row = new Rect(64,238+i*rowHeight,802,rowHeight-4); var o=options[i];
                var mouse = new Vector2(Input.mousePosition.x,Input.mousePosition.y);
                if (mouse != lastMouse && Hover(row)) navigation.PointAt(i);
                var selected = navigation.Focus == i;
                Fill(row, selected ? new Color(0,.19f,.23f,.98f) : new Color(0,0,.1f,.94f));
                if (selected) Fill(new Rect(row.x,row.y,4,row.height),gold);
                Text(row.x+18,row.y,382,row.height,o.name,18,o.enabled?paper:dim,true);
                Text(row.x+396,row.y,280,row.height,o.value,17,o.enabled?gold:dim,true,TextAnchor.MiddleRight);
                if (o.enabled)
                {
                    if (GUI.Button(new Rect(row.x+746,row.y,52,row.height),">",hit)) { navigation.PointAt(i); o.edit(1); }
                    if (GUI.Button(new Rect(row.x+696,row.y,45,row.height),"<",hit)) { navigation.PointAt(i); o.edit(-1); }
                }
            }
            lastMouse = new Vector2(Input.mousePosition.x,Input.mousePosition.y);
            var footer = new[] { "DEFAULTS", "SCREENSHOT", "RESTART CAVE", "BACK" };
            for (var i=0;i<4;i++)
            {
                var rect=new Rect(64+i*201,858,190,48); var enabled=i!=2||game.Connected&&game.HasExpedition&&S.realm!="mine"&&!game.OpeningCave;
                Fill(rect,navigation.Focus==options.Count+i?gold:new Color(0,0,.33f));
                Text(rect.x+12,rect.y,rect.width-24,rect.height,footer[i],17,enabled?navigation.Focus==options.Count+i?ink:paper:dim,true);
                if (enabled && GUI.Button(rect,GUIContent.none,hit)) { navigation.PointAt(options.Count+i); FooterAction(i); }
            }
            Text(64,920,802,25,navigation.Controller?"STICK  SELECT / ADJUST    A  APPLY    B  BACK    LB/RB  TABS":"↑ ↓  SELECT    ← →  ADJUST    ENTER  APPLY    ESC  BACK    TAB  TABS",15,dim);
            PixelText(940,152,"LIVE CAVE PREVIEW",3,gold);
            Text(940,197,606,54,"Your expedition is paused. Changes appear here immediately.",21,paper);
            Plate(new Rect(938,700,612,225));
            var help=navigation.Focus<options.Count?options[navigation.Focus].help:"Defaults reset presentation and audio. Screenshot mode pauses the cave and lets you hide the HUD. Restart starts this cave again.";
            Text(962,723,565,145,help,22,paper);
            if (settingsTab==1)
            {
                for(var i=0;i<5;i++)
                {
                    var value=(.04f+i*.04f)*CaveSettings.Data.brightness/100f;
                    var tint=CaveSettings.Data.palette==1?new Color(1.08f,1,.92f):CaveSettings.Data.palette==2?new Color(.94f,1,1.08f):Color.white;
                    Fill(new Rect(965+i*106,870,94,28),new Color(value*tint.r,value*tint.g,value*tint.b));
                }
            }
            if (game.Presentation.Pending) DisplayConfirmation();
        }
        void DisplayConfirmation()
        {
            Scrim(.82f); Plate(new Rect(425,340,750,290));
            PixelText(465,375,"KEEP THIS DISPLAY?",4,gold);
            Text(465,431,650,65,"Reverting in "+Mathf.CeilToInt(game.Presentation.SecondsLeft)+" seconds.\nEnter / A keeps it. Escape / B restores the previous display.",22,paper);
            if(Button(465,539,280,58,"KEEP",true,true))game.Presentation.Keep();
            if(Button(775,539,350,58,"REVERT"))game.Presentation.Revert();
        }
        void ReadableHud(float y)
        {
            var d=CaveSettings.Data; var large=d.hudScale>1; var scale=large?3:2; var margin=d.safeMargin+8;
            var tint=d.highContrast?Color.white:new Color(1f/3,1,1f/3);
            PixelText(margin,y+(large?16:10),"$"+S.score.ToString("D6"),scale,tint);
            var x=margin+(large?156:130); var icon=large?30:22;
            Image(new Rect(x,y+6,icon,icon),Resources.Load<Texture2D>(S.realm=="mine"?"Interface/crystal":"Sprites/"+CrystalName));
            PixelText(x+icon+6,y+(large?16:10),S.realm=="mine"?S.cleared_caves+"/16":S.crystals+"/"+S.initial_crystals,scale,d.highContrast?paper:gold);
            x=margin+(large?300:252);
            Image(new Rect(x,y+6,large?39:26,large?33:22),Resources.Load<Texture2D>("Interface/raygun"));
            PixelText(x+(large?45:34),y+(large?16:10),S.ammo.ToString(),scale,tint);
            x=margin+(large?410:360);
            for(var i=0;i<3;i++)Image(new Rect(x+i*(large?31:21),y+(large?10:8),large?27:18,large?24:16),heart,S.realm=="mine"||S.health>i?paper:new Color(.25f,.25f,.25f));
            if(large)PixelText(margin+410,y+36,S.realm=="mine"?"MAIN MINE":"CAVE "+(S.level+1).ToString("00"),1,paper);
            if(!large)PixelText(margin+440,y+12,S.realm=="mine"?"MINE":"CAVE "+(S.level+1).ToString("00"),2,paper);
            if(S.realm=="mine")
            {
                if(S.near_entrance>=0)
                {
                    Fill(new Rect(150,CaveViewport.HintY,340,23),new Color(0,0,0,.94f));
                    PixelText(160,CaveViewport.HintY+5,"CAVE "+(S.near_entrance+1).ToString("00")+" / "+S.levels[S.near_entrance]+"   [E / ENTER]",1,gold);
                }
                else PixelText(180,CaveViewport.WorldTop+10,"FIND A DOOR / UP-DOWN TO CLIMB CHAINS",1,dim);
                foreach(var entrance in S.entities)
                {
                    if(!entrance.id.StartsWith("entrance_"))continue;
                    var point=CaveViewport.Point(game.GameCamera.WorldToScreenPoint(CaveWorld.Position(entrance.x+7,entrance.y-10)));
                    if(point.x>0&&point.x<CaveViewport.Width-16&&point.y>CaveViewport.WorldTop+3&&point.y<CaveViewport.WorldBottom-8)
                        PixelText(point.x,point.y,(int.Parse(entrance.id.Substring(9))+1).ToString("00"),1,gold);
                }
            }
            else if(S.mode=="ai")PixelText(12,CaveViewport.WorldTop+12,"AI AT THE CONTROLS / F2",1,teal);
        }
        void DrawItemMarkers()
        {
            foreach(var entity in S.entities)
            {
                string marker=null;
                if(entity.id.StartsWith("crystal_")) { var index=(Mathf.RoundToInt(entity.x/32)+Mathf.RoundToInt(entity.y/32)*3)%4; marker=new[]{"B","G","Y","R"}[index]; }
                else if(entity.sprite.Contains("heart") || entity.sprite.Contains("health"))marker="+";
                if(marker==null)continue;
                var p=CaveViewport.Point(game.GameCamera.WorldToScreenPoint(CaveWorld.Position(entity.x+entity.w/2,entity.y-7)));
                if(p.x<6||p.x>CaveViewport.Width-12||p.y<CaveViewport.WorldTop+8||p.y>CaveViewport.WorldBottom-10)continue;
                Fill(new Rect(p.x-4,p.y-2,8,12),Color.black); PixelText(p.x-3,p.y,marker,1,paper);
            }
        }
        void DrawPresentationOverlay()
        {
            if(game.Presentation.Pending&&game.Screen!=CaveScreen.Options)DisplayConfirmation();
            if(game.ScreenshotMode)
            {
                if(!game.CapturingScreenshot)
                {
                    Plate(new Rect(350,840,900,105));
                    Text(378,851,850,78,"SCREENSHOT MODE    F12 / A  SAVE    H / X  HUD    ESC / B  BACK\n"+(game.HideHud?"HUD hidden":"HUD visible"),20,paper,true,TextAnchor.MiddleCenter);
                }
            }
            if(CaveSettings.Data.showFps&&!game.CapturingScreenshot) { Plate(new Rect(1390,14,192,40)); Text(1400,14,170,40,Mathf.RoundToInt(game.Presentation.Fps)+" FPS",18,paper,true,TextAnchor.MiddleRight); }
            if(Time.unscaledTime<game.ScreenshotToastUntil&&!game.CapturingScreenshot)
            { Plate(new Rect(260,30,1080,86)); Text(280,37,1040,72,game.ScreenshotMessage,19,paper,true); }
        }
    }
}

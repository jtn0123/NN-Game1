using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace CrystalCaves.Pilot
{
    public sealed partial class CavePilot
    {
        bool settingsSmoke;
        readonly List<string> settingsChecks = new List<string>();
        readonly List<string> settingsSkipped = new List<string>();
        [Serializable] sealed class SettingsReport { public bool success; public string error, screenshot; public string[] checks, skipped; public int width,height; }
        void CheckSettings(bool success,string message)
        { if(!success)throw new InvalidOperationException(message); settingsChecks.Add(message); Debug.Log("Settings check: "+message); }
        bool TryStartSettingsSmoke(string[] arguments)
        {
            var index=Array.IndexOf(arguments,"--settings-smoke");if(index<0)return false;
            if(index+1>=arguments.Length)throw new ArgumentException("--settings-smoke needs an output directory");
            var output=arguments[index+1];Directory.CreateDirectory(output);
            var stateIndex=Array.IndexOf(arguments,"--settings-state");
            if(stateIndex>=0&&stateIndex+1<arguments.Length)state=JsonUtility.FromJson<CaveSnapshot>(File.ReadAllText(arguments[stateIndex+1]));
            var caseIndex=Array.IndexOf(arguments,"--settings-case");var scenario=caseIndex>=0&&caseIndex+1<arguments.Length?arguments[caseIndex+1]:"menus";
            settingsSmoke=true;HasExpedition=true;Screen=CaveScreen.Play;
            UnityEngine.Rendering.SplashScreen.Stop(UnityEngine.Rendering.SplashScreen.StopBehavior.StopImmediate);
            world.Apply(state);StartCoroutine(RunSettingsSmoke(output,scenario));return true;
        }
        IEnumerator RunSettingsSmoke(string output,string scenario)
        {
            var routine=SettingsScenario(output,scenario);Exception failure=null;
            while(true)
            {
                bool next=false;try{next=routine.MoveNext();}catch(Exception error){failure=error;}
                if(failure!=null||!next)break;
                yield return routine.Current;
            }
            presentation.Revert();CaveViewport.Preview=CaveViewport.HideHudStrip=false;CaveVisualClock.Frozen=false;
            var report=new SettingsReport { success=failure==null,error=failure?.ToString(),screenshot=ScreenshotMessage,checks=settingsChecks.ToArray(),skipped=settingsSkipped.ToArray(),width=UnityEngine.Screen.width,height=UnityEngine.Screen.height };
            File.WriteAllText(Path.Combine(output,"settings-report.json"),JsonUtility.ToJson(report,true));
            if(failure!=null)Debug.LogException(failure);Application.Quit(failure==null?0:1);
        }
        IEnumerator SettingsScenario(string output,string scenario)
        {
            yield return null;
            if(scenario=="menus")
            {
                Open(CaveScreen.Options);
                for(var tab=0;tab<4;tab++) {view.SetSettingsTab(tab);yield return SettingsImage(output,"settings-"+new[]{"display","graphics","sound","accessibility"}[tab]);}
                var step=state.steps;yield return new WaitForSecondsRealtime(.25f);CheckSettings(state.steps==step,"Settings preview keeps simulation paused");
                view.SetSettingsTab(1);var shake=CaveSettings.Data.shake;
                view.Navigate(1,-1,false,false,0);CheckSettings(CaveSettings.Data.shake==shake-10,"Focused graphics row responds to keyboard/controller adjustment");
                view.Navigate(0,0,false,false,1);CheckSettings(view.SettingsTab==2,"Shoulder/Tab navigation changes pages");
                view.Navigate(0,0,false,true,0);CheckSettings(Screen==CaveScreen.Play,"Back returns to the originating screen");
                Open(CaveScreen.Options);CaveSettings.Change(d=>d.screenshotHideHud=true,false);StartScreenshotMode();
                CheckSettings(ScreenshotMode&&HideHud&&CaveViewport.HudHeight==0,"Screenshot mode can hide HUD and reclaim the camera strip");
                yield return SettingsImage(output,"screenshot-mode");SaveScreenshot();
                while(CapturingScreenshot)yield return null;
                CheckSettings(ScreenshotMessage.StartsWith("Screenshot saved")&&Directory.GetFiles(ScreenshotDirectory,"*.png").Length>0,"Screenshot action writes a PNG and reports its path");
                yield return SettingsImage(output,"screenshot-saved");ExitScreenshotMode();CheckSettings(Screen==CaveScreen.Options&&!HideHud&&!CaveVisualClock.Frozen,"Screenshot back restores options and the normal HUD");
                CaveSettings.Change(d=>{d.masterVolume=40;d.effectsVolume=50;d.muted=false;},false);
                CheckSettings(Mathf.Abs(audioPlayer.Volume-.4f)<.001f,"Sound settings update the live audio player");
                audioPlayer.SetPlaying(false); audioPlayer.Preview("shoot");CheckSettings(audioPlayer.CurrentSound=="shoot" && audioPlayer.IsPlaying,"Sound preview plays the actual raygun cue while settings are paused");
            }
            else if(scenario=="vsync")
            {
                yield return new WaitForSecondsRealtime(.8f);
                CheckSettings(presentation.Fps>0,"Native player advances without V-sync");
                CaveSettings.Change(d=>d.vSync=true,false);
                yield return new WaitForSecondsRealtime(1f);
                CheckSettings((CaveNativeDisplay.IsMacPlayer?CaveNativeDisplay.Status==2:QualitySettings.vSyncCount==1)&&presentation.Fps>0,"V-sync advances native frames after initialization");
                yield return SettingsImage(output,"vsync-native");
                CaveSettings.Change(d=>d.vSync=false,false);
                yield return new WaitForSecondsRealtime(.25f);
                CheckSettings(QualitySettings.vSyncCount==0 && (!CaveNativeDisplay.IsMacPlayer || CaveNativeDisplay.Status==1),"V-sync can return to capped presentation");
                CaveSettings.Change(d=>d.frameCap=30,false);yield return new WaitForSecondsRealtime(1.2f);
                CheckSettings(Mathf.Abs(presentation.Fps-30)<8,"Native frame cap renders near 30 FPS");
                CaveSettings.Change(d=>d.frameCap=60,false);yield return new WaitForSecondsRealtime(1.2f);
                CheckSettings(Mathf.Abs(presentation.Fps-60)<8,"Native frame cap renders near 60 FPS");
            }
            else if(scenario=="display")
            {
                var width=UnityEngine.Screen.width;var height=UnityEngine.Screen.height;
                Open(CaveScreen.Options);presentation.ApplyDisplay(960,600,false);yield return new WaitForSecondsRealtime(.8f);
                CheckSettings(presentation.Pending&&UnityEngine.Screen.width==960&&UnityEngine.Screen.height==600,"Window size applies with confirmation pending");
                yield return SettingsImage(output,"display-confirmation");presentation.Revert();yield return new WaitForSecondsRealtime(.8f);
                CheckSettings(!presentation.Pending&&UnityEngine.Screen.width==width&&UnityEngine.Screen.height==height,"Explicit revert restores the prior window dimensions");
                presentation.ApplyDisplay(960,600,false);yield return new WaitForSecondsRealtime(15.8f);
                CheckSettings(!presentation.Pending&&UnityEngine.Screen.width==width&&UnityEngine.Screen.height==height,"Unconfirmed display automatically reverts after 15 seconds");
                if(CaveNativeDisplay.WindowActive)
                {
                    presentation.ApplyDisplay(960,600,true);
                    var enterDeadline=Time.realtimeSinceStartup+5;
                    while(Time.realtimeSinceStartup<enterDeadline && (!UnityEngine.Screen.fullScreen || !CaveNativeDisplay.WindowMatchesSize(UnityEngine.Screen.width,UnityEngine.Screen.height)))yield return null;
                    CheckSettings(UnityEngine.Screen.fullScreen && presentation.Pending && CaveNativeDisplay.WindowMatchesSize(UnityEngine.Screen.width,UnityEngine.Screen.height),"Fullscreen applies at the display's native size with confirmation");
                    presentation.Revert();
                    var restoreDeadline=Time.realtimeSinceStartup+5;
                    while(Time.realtimeSinceStartup<restoreDeadline && (UnityEngine.Screen.fullScreen || UnityEngine.Screen.width!=width || UnityEngine.Screen.height!=height))yield return null;
                    Debug.Log("Fullscreen restore observed mode="+UnityEngine.Screen.fullScreenMode+" native="+CaveNativeDisplay.WindowState);
                    CheckSettings(!UnityEngine.Screen.fullScreen && UnityEngine.Screen.width==width && UnityEngine.Screen.height==height && CaveNativeDisplay.WindowMatchesSize(width,height),"Fullscreen revert restores the original native window");
                }
                else
                {
                    const string reason="Fullscreen apply/revert requires an active native desktop; app inactive (loginwindow observed by runner)";
                    settingsSkipped.Add(reason);Debug.LogWarning(reason);
                }
                presentation.ApplyDisplay(960,600,false);yield return new WaitForSecondsRealtime(.8f);presentation.Keep();
                CheckSettings(!presentation.Pending&&CaveSettings.Data.width==960&&CaveSettings.Data.height==600,"Keep commits the requested window size");
            }
            else if(scenario=="render")
            {
                CaveSettings.Change(d=>d.ApplyPreset(0),false);Screen=CaveScreen.Play;
                yield return SettingsImage(output,"current-retro");
                CaveSettings.Change(d=>d.ApplyPreset(1),false);yield return SettingsImage(output,"enhanced-retro");
                CaveSettings.Change(d=>{d.scanlines=30;d.phosphor=20;d.brightness=110;d.palette=2;});yield return SettingsImage(output,"crt-cool");
                CheckSettings(postFx.enabled,"World-only CRT/brightness/palette pass is active");
                CaveSettings.Change(d=>{d.hudScale=2;d.safeMargin=12;d.highContrast=true;d.itemMarkers=true;},false);yield return SettingsImage(output,"readable-hud");
                CheckSettings(CaveViewport.HudHeight==48&&Mathf.Abs(gameCamera.pixelRect.height-(CaveViewport.Frame.height-48*CaveViewport.Scale))<1,"Large HUD reserves its own camera strip");
                CaveSettings.Change(d=>d.ApplyPreset(2),false);yield return SettingsImage(output,"low-power");
                CaveSettings.Change(d=>{d.reducedMotion=true;d.damageFlashes=false;},false);world.Apply(state);world.Animate(state,0);
                var enemy=Array.Find(state.entities,e=>e.sprite=="dinosaur_enemy");
                if(enemy!=null)
                {
                    var renderer=GameObject.Find(enemy.id).GetComponent<SpriteRenderer>();
                    var pose=enemy.sprite+(enemy.hit?"_hit":"")+"_"+((state.steps+state.freeze_timer)/6%4);
                    CheckSettings(renderer.sprite.texture.name==pose,"Reduced motion keeps the creature's actual movement pose");
                }
                var lift=Array.Find(state.entities,e=>e.id.StartsWith("lift_"));
                if(lift!=null)CheckSettings(GameObject.Find(lift.id).GetComponent<SpriteRenderer>().sprite.texture.name=="elevator_"+lift.frame,"Reduced motion retains lift animation");
                if(state.player.invulnerable)CheckSettings(GameObject.Find("player").GetComponent<SpriteRenderer>().sprite.texture.name.EndsWith("_hit"),"Disabled damage flashes use a steady hurt pose");
                CheckSettings(state.steps>0&&state.player!=null,"Reduced motion retains the authoritative gameplay snapshot");
                CaveSettings.Change(d=>{d.pixelScale=6;d.aspect=1;d.hudScale=1;},false);world.Animate(state,0);
                CheckSettings(CaveViewport.Scale<=UnityEngine.Screen.width/640&&CaveViewport.Height<=400,"Oversized integer scaling safely fits the window");
                CaveSettings.Change(d=>{d.pixelScale=1;d.aspect=2;},false);world.Animate(state,0);
                CheckSettings(CaveViewport.Width==UnityEngine.Screen.width,"Wide framing fills width without stretching pixel art");
            }
            else throw new ArgumentException("Unknown settings scenario: "+scenario);
        }
        IEnumerator SettingsImage(string output,string name)
        {
            CaveViewport.Preview=Screen==CaveScreen.Options;postFx.enabled=CavePostFx.Needed;
            world.Apply(state);world.Animate(state,0);
            // Let both layout and repaint settle after changing a page or viewport.
            yield return null;
            yield return new WaitForEndOfFrame();
            var image=ScreenCapture.CaptureScreenshotAsTexture();try{File.WriteAllBytes(Path.Combine(output,name+".png"),image.EncodeToPNG());}finally{Destroy(image);}
        }
    }
}

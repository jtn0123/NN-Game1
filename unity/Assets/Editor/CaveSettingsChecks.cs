using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace CrystalCaves.Pilot.Editor
{
    public static class CaveSettingsChecks
    {
        [Serializable] sealed class Report { public bool success; public string[] checks; }
        public static void Run()
        {
            var checks = new List<string>();
            Action<bool,string> check = (ok,name) => { if(!ok)throw new Exception("Settings check failed: "+name); checks.Add(name); };
            var stored = PlayerPrefs.GetString(CaveSettings.Key,"__missing__");
            var d = new CaveSettingsData();
            check(d.preset=="Current Retro" && d.lighting==0 && !d.environment && d.brightness==100 && d.scanlines==0 && d.phosphor==0,"Retro defaults preserve the original presentation");
            d.width=99999;d.height=-1;d.pixelScale=999;d.hudScale=0;d.brightness=0;d.effectDensity=-4;d.frameCap=12;d.masterVolume=200;d.palette=10;d.shadowStrength=99;d.preset="bad";d.Sanitize();
            check(d.width==7680 && d.height==384 && d.pixelScale==6 && d.hudScale==1 && d.brightness==70 && d.effectDensity==0 && d.frameCap==30 && d.masterVolume==100 && d.palette==2 && d.shadowStrength==80 && d.preset=="Custom","Corrupt/out-of-range values are bounded");
            d=new CaveSettingsData { width=1280,height=800,fullscreen=true,hudBottom=false,reducedMotion=true,masterVolume=35,safeMargin=12,hudScale=2,highContrast=true,itemMarkers=true,screenshotHideHud=true };
            var copy=CaveSettings.Parse(JsonUtility.ToJson(d));
            check(JsonUtility.ToJson(copy)==JsonUtility.ToJson(d),"All settings round-trip through JSON");
            var legacy=CaveSettings.FromLegacy(1,1,.45f,1);
            check(!legacy.hudBottom && legacy.reducedMotion && legacy.masterVolume==45 && legacy.muted,"Legacy HUD, motion and audio preferences migrate");
            check(CaveSettings.Parse("{").brightness==100,"Malformed saved data falls back safely");
            d.ApplyPreset(1);
            check(d.lighting==25 && d.environment && d.width==1280 && d.fullscreen && !d.hudBottom && d.reducedMotion && d.masterVolume==35 && d.hudScale==2 && d.safeMargin==12,"Enhanced preset preserves display/audio/accessibility");
            d.ApplyPreset(2);check(d.frameCap==30 && !d.vSync && d.effectDensity==0 && !d.contactShadows && !d.gemGlints,"Low Power reduces presentation work");
            d.ApplyPreset(0);check(d.lighting==0 && !d.environment && d.effectDensity==2 && d.contactShadows && !d.vSync && d.frameCap==60,"Current Retro restores baseline effects");
            CaveSettings.Load(true,JsonUtility.ToJson(d));CaveSettings.Change(v=>v.scanlines=20);
            check(CaveSettings.Data.preset=="Custom","Manual overrides mark the preset Custom");
            CaveSettings.Change(v=>{v.vSync=false;v.frameCap=90;});check(QualitySettings.vSyncCount==0 && Application.targetFrameRate==90,"Frame cap applies with V-sync disabled");
            CaveSettings.Change(v=>v.vSync=true);check(QualitySettings.vSyncCount==1 && Application.targetFrameRate==-1,"V-sync controls presentation pacing");
            CaveSettings.Change(v=>{v.vSync=false;v.frameCap=0;});check(Application.targetFrameRate==-1,"Unlimited frame cap works");
            check(PlayerPrefs.GetString(CaveSettings.Key,"__missing__")==stored,"Automated review never writes player preferences");
            var nav=new CaveMenuNavigation();nav.Apply(5,-1,0,false,false,0);check(nav.Focus==4,"Navigation wraps backward");
            nav.Apply(5,1,1,true,false,0);check(nav.Focus==0 && nav.Adjustment(0)==1 && nav.Activate(0) && !nav.Activate(0),"Navigation adjusts and consumes a submit once");
            nav.Apply(5,0,0,false,true,-1);check(nav.Back&&nav.TabDelta==-1,"Controller/keyboard back and tab actions share navigation");
            var args=Environment.GetCommandLineArgs();var index=Array.IndexOf(args,"--settings-check-report");
            if(index>=0&&index+1<args.Length)File.WriteAllText(args[index+1],JsonUtility.ToJson(new Report { success=true,checks=checks.ToArray() },true));
            Debug.Log("Passed "+checks.Count+" settings behavior checks");
        }
    }
}

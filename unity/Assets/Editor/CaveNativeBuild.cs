using System;
using System.Diagnostics;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace CrystalCaves.Pilot.Editor
{
    public static class CaveNativeBuild
    {
        public static void Compile()
        {
            var source = Path.GetFullPath(Path.Combine(Application.dataPath,"../Native/CaveDisplay.mm"));
            const string target="Assets/Plugins/macOS/libCaveDisplay.dylib";
            Directory.CreateDirectory(Path.GetDirectoryName(target));
            var start=new ProcessStartInfo("/usr/bin/xcrun")
            {
                Arguments="clang++ -dynamiclib -fobjc-arc -arch arm64 -arch x86_64 -mmacosx-version-min=11.0 -framework AppKit -framework QuartzCore -framework Metal -install_name @rpath/libCaveDisplay.dylib \""+source+"\" -o \""+Path.GetFullPath(target)+"\"",
                UseShellExecute=false,RedirectStandardError=true,CreateNoWindow=true
            };
            using(var process=Process.Start(start))
            {
                var error=process.StandardError.ReadToEnd();process.WaitForExit();
                if(process.ExitCode!=0)throw new Exception("Native display plugin failed: "+error);
            }
            AssetDatabase.ImportAsset(target,ImportAssetOptions.ForceSynchronousImport);
            var importer=(PluginImporter)AssetImporter.GetAtPath(target);
            importer.SetCompatibleWithAnyPlatform(false);importer.SetCompatibleWithEditor(false);
            importer.SetCompatibleWithPlatform(BuildTarget.StandaloneOSX,true);importer.SaveAndReimport();
        }
    }
}

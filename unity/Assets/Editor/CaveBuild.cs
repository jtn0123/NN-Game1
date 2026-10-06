using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace CrystalCaves.Pilot.Editor
{
    public sealed class CaveAudioImporter : AssetPostprocessor
    {
        void OnPreprocessAudio()
        {
            if (!assetPath.StartsWith("Assets/Resources/Audio/")) return;
            var importer = (AudioImporter)assetImporter;
            var settings = importer.defaultSampleSettings;
            settings.compressionFormat = AudioCompressionFormat.PCM;
            settings.loadType = AudioClipLoadType.DecompressOnLoad;
            settings.sampleRateSetting = AudioSampleRateSetting.PreserveSampleRate;
            importer.defaultSampleSettings = settings;
            importer.forceToMono = true;
        }
    }

    public sealed class CaveTextureImporter : AssetPostprocessor
    {
        void OnPreprocessTexture()
        {
            if (!assetPath.StartsWith("Assets/Resources/Sprites/") && !assetPath.StartsWith("Assets/Resources/Terrain/") && !assetPath.StartsWith("Assets/Resources/Interface/") && !assetPath.StartsWith("Assets/Resources/Environment/")) return;
            var importer = (TextureImporter)assetImporter;
            importer.textureType = TextureImporterType.Default;
            importer.filterMode = FilterMode.Point;
            importer.mipmapEnabled = false;
            importer.npotScale = TextureImporterNPOTScale.None;
            importer.alphaIsTransparency = true;
            importer.textureCompression = TextureImporterCompression.Uncompressed;
            importer.maxTextureSize = 2048;
        }
    }

    public static class CaveBuild
    {
        [MenuItem("Crystal Caves/Prepare pilot scene")]
        public static void Prepare()
        {
            foreach (var guid in AssetDatabase.FindAssets("t:Texture2D", new[] { "Assets/Resources/Terrain", "Assets/Resources/Sprites", "Assets/Resources/Interface", "Assets/Resources/Environment" }))
            {
                var importer = (TextureImporter)AssetImporter.GetAtPath(AssetDatabase.GUIDToAssetPath(guid));
                if (importer.npotScale != TextureImporterNPOTScale.None || importer.filterMode != FilterMode.Point || importer.maxTextureSize != 2048 || importer.mipmapEnabled)
                {
                    importer.npotScale = TextureImporterNPOTScale.None;
                    importer.filterMode = FilterMode.Point;
                    importer.maxTextureSize = 2048;
                    importer.mipmapEnabled = false;
                    importer.SaveAndReimport();
                }
            }
            Directory.CreateDirectory("Assets/Scenes");
            if (!File.Exists("Assets/Scenes/CrystalCaves.unity"))
            {
                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                EditorSceneManager.SaveScene(scene, "Assets/Scenes/CrystalCaves.unity");
            }
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene("Assets/Scenes/CrystalCaves.unity", true) };
            PlayerSettings.companyName = "NN Game1";
            PlayerSettings.productName = "Crystal Caves";
            PlayerSettings.defaultScreenWidth = 1600;
            PlayerSettings.defaultScreenHeight = 1000;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.resizableWindow = true;
            PlayerSettings.runInBackground = true;
            PlayerSettings.colorSpace = ColorSpace.Gamma;
            PlayerSettings.SetApplicationIdentifier(UnityEditor.Build.NamedBuildTarget.Standalone, "com.nngame1.crystalcaves");
            AssetDatabase.SaveAssets();
            Debug.Log("Crystal Caves pilot scene prepared");
        }

        [MenuItem("Crystal Caves/Build macOS pilot")]
        public static void Build()
        {
            CaveNativeBuild.Compile();
            Prepare();
            Directory.CreateDirectory("Builds");
            var options = new BuildPlayerOptions
            {
                scenes = new[] { "Assets/Scenes/CrystalCaves.unity" },
                locationPathName = "Builds/Crystal Caves.app",
                target = BuildTarget.StandaloneOSX,
                options = BuildOptions.None
            };
            var report = BuildPipeline.BuildPlayer(options);
            if (report.summary.result != BuildResult.Succeeded)
                throw new Exception("Crystal Caves pilot build failed: " + report.summary.result);
            Debug.Log("Crystal Caves pilot built: " + report.summary.outputPath);
        }
    }
}

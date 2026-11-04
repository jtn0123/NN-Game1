using System;
using System.Reflection;
using System.Runtime.ExceptionServices;
using NUnit.Framework;
using UnityEngine;

namespace CrystalCaves.Pilot.Tests
{
    public sealed class EditorContractTests
    {
        int vSyncCount;
        int targetFrameRate;

        [SetUp]
        public void SaveFramePacing()
        {
            vSyncCount = QualitySettings.vSyncCount;
            targetFrameRate = Application.targetFrameRate;
        }

        [TearDown]
        public void RestoreFramePacing()
        {
            QualitySettings.vSyncCount = vSyncCount;
            Application.targetFrameRate = targetFrameRate;
        }

        [TestCase("CaveControlsChecks")]
        [TestCase("CavePowerFeedbackChecks")]
        [TestCase("CaveSettingsChecks")]
        [TestCase("CaveSnapshotReaderChecks")]
        [TestCase("CaveLocalBackendChecks")]
        public void ExistingEditorChecksPass(string suite)
        {
            // Test asmdefs cannot reference Unity's predefined editor assembly.
            // Keep the existing menu/batch entry points there and invoke them
            // without moving the game's runtime assemblies into the test graph.
            var type = Type.GetType("CrystalCaves.Pilot.Editor." + suite + ", Assembly-CSharp-Editor", true);
            var run = type.GetMethod("Run", BindingFlags.Public | BindingFlags.Static);
            Assert.That(run, Is.Not.Null, suite + " must expose public static Run()");
            try { run.Invoke(null, null); }
            catch (TargetInvocationException exception)
            {
                ExceptionDispatchInfo.Capture(exception.InnerException ?? exception).Throw();
            }
        }
    }
}

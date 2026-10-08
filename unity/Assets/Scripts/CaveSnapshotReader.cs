using System;
using UnityEngine;

namespace CrystalCaves.Pilot
{
    // Parsing has no scene, input, transport or expedition state to leave half-open.
    public static class CaveSnapshotReader
    {
        public static bool TryRead(string json, out CaveSnapshot snapshot)
        {
            snapshot = null;
            var candidate = new CaveSnapshot();
            try { JsonUtility.FromJsonOverwrite(json, candidate); }
            catch (ArgumentException) { return false; }
            if (candidate.protocol != 1) return false;
            // A server error intentionally has no player and still belongs to
            // CavePilot's error branch rather than malformed-response recovery.
            if (!string.IsNullOrEmpty(candidate.error))
            {
                candidate.player = null;
                snapshot = candidate;
                return true;
            }
            // JsonUtility invents nested classes for absent/null fields, even
            // with Overwrite. Require real player and room data before Apply.
            if (candidate.player == null || string.IsNullOrEmpty(candidate.player.sprite)
                || candidate.cols <= 0 || candidate.rows <= 0 || candidate.layout == null
                || candidate.layout.Length != candidate.rows
                || candidate.entities == null || candidate.effects == null) return false;
            foreach (var row in candidate.layout)
                if (row == null || row.Length != candidate.cols) return false;
            snapshot = candidate;
            return true;
        }
    }
}

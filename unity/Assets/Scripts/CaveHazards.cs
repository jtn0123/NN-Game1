using System.Collections.Generic;
using UnityEngine;

namespace CrystalCaves.Pilot
{
    public sealed class CaveHazards
    {
        readonly Transform root;
        readonly List<SpriteRenderer> surfaces = new List<SpriteRenderer>();
        readonly CaveParticles particles;
        readonly Sprite wave;
        float vaporTime;

        public CaveHazards(Transform parent, CaveParticles particles)
        {
            this.particles = particles;
            root = new GameObject("Living acid surfaces").transform; root.SetParent(parent);
            var texture = new Texture2D(32, 8, TextureFormat.RGBA32, false) { filterMode = FilterMode.Point };
            for (var y = 0; y < 8; y++) for (var x = 0; x < 32; x++)
                texture.SetPixel(x, y, y == (x / 5 % 2 + 2) ? new Color(1f / 3, 1, 1f / 3) : Color.clear);
            texture.Apply(); wave = Sprite.Create(texture, new Rect(0, 0, 32, 8), Vector2.one * .5f, 32);
        }
        public void Build(CaveSnapshot state)
        {
            foreach (var surface in surfaces) Object.Destroy(surface.gameObject); surfaces.Clear();
            for (var row = 0; row < state.rows; row++) for (var col = 0; col < state.cols; col++)
            {
                if (state.layout[row][col] != '~') continue;
                var obj = new GameObject("Acid surface", typeof(SpriteRenderer)); obj.transform.SetParent(root);
                var renderer = obj.GetComponent<SpriteRenderer>(); renderer.sprite = wave; renderer.sortingOrder = 2;
                renderer.transform.position = CaveWorld.Position(col * 32 + 16, row * 32 + 7);
                surfaces.Add(renderer);
            }
        }
        public void Animate(Camera camera, float delta)
        {
            foreach (var surface in surfaces)
            {
                surface.flipX = ((int)(CaveVisualClock.Now * 4) + (int)surface.transform.position.x) % 2 == 0;
            }
            if (!CaveSettings.Data.environment || !CaveVisualSettings.Motion) return;
            vaporTime += delta;
            if (vaporTime < 1.1f || surfaces.Count == 0) return;
            vaporTime = 0;
            var surfaceIndex = (int)(CaveVisualClock.Now * 3) % surfaces.Count;
            var point = surfaces[surfaceIndex].transform.position;
            if (Vector3.Distance(point, camera.transform.position + Vector3.forward * 10) < 12)
                particles.Burst(point.x * 32, -point.y * 32, "vapor");
        }
    }
}

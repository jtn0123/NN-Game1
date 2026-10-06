using UnityEngine;

namespace CrystalCaves.Pilot
{
    public sealed class CaveParticles
    {
        sealed class Particle
        {
            public SpriteRenderer renderer;
            public Vector3 position, velocity;
            public Color color;
            public float remaining;
        }
        readonly Particle[] pool = new Particle[96];
        readonly Sprite pixel;
        int cursor;

        public CaveParticles(Transform parent, Sprite unused)
        {
            var texture = new Texture2D(1, 1, TextureFormat.RGBA32, false) { filterMode = FilterMode.Point };
            texture.SetPixel(0, 0, Color.white); texture.Apply();
            pixel = Sprite.Create(texture, new Rect(0, 0, 1, 1), Vector2.one * .5f, 1);
            for (var i = 0; i < pool.Length; i++)
            {
                var instance = new GameObject("Pixel feedback", typeof(SpriteRenderer)); instance.transform.SetParent(parent, false);
                var renderer = instance.GetComponent<SpriteRenderer>(); renderer.sortingOrder = 7; renderer.enabled = false;
                pool[i] = new Particle { renderer = renderer };
            }
        }

        public void Burst(float x, float y, string kind, int facing = 1)
        {
            if (!CaveVisualSettings.Motion || !CaveSettings.Data.particles || CaveSettings.Data.effectDensity == 0) return;
            var shot = kind == "shot" || kind == "spark";
            var pickup = kind == "sparkle" || kind == "score";
            var damage = kind == "damage";
            var vapor = kind == "vapor";
            var count = vapor ? 2 : shot ? 5 : pickup ? 8 : 6;
            if (CaveSettings.Data.effectDensity == 1) count = Mathf.Max(1, count / 2);
            var tint = damage ? new Color(1, 1f / 3, 1f / 3) : pickup ? Color.white : vapor ? new Color(1f / 3, 1, 1f / 3) : shot ? new Color(1, 1, 1f / 3) : new Color(2f / 3, 2f / 3, 2f / 3);
            for (var i = 0; i < count; i++)
            {
                var angle = i * 2.399963f;
                var speed = pickup ? 1 + i % 3 * .4f : shot ? 2 + i % 3 : .5f + i % 3 * .25f;
                var p = pool[cursor++ % pool.Length];
                p.position = CaveWorld.Position(x, y);
                p.velocity = new Vector3(Mathf.Cos(angle) * speed, Mathf.Sin(angle) * speed + .5f, 0);
                if (shot) p.velocity = new Vector3(facing * speed, Mathf.Sin(angle) * .45f, 0);
                if (vapor) p.velocity = new Vector3(Mathf.Sin(angle) * .1f, .3f, 0);
                p.remaining = vapor ? .8f : .25f + i % 3 * .1f;
                p.color = tint; p.renderer.sprite = pixel; p.renderer.color = tint;
                p.renderer.transform.position = p.position;
                p.renderer.transform.localScale = shot ? new Vector3(.1875f, .0625f, 1) : Vector3.one * .0625f;
                p.renderer.transform.rotation = Quaternion.identity;
                p.renderer.enabled = true;
            }
        }

        public void Animate(float delta)
        {
            foreach (var p in pool)
            {
                if (!CaveVisualSettings.Motion || !CaveSettings.Data.particles || CaveSettings.Data.effectDensity == 0) { p.remaining = 0; p.renderer.enabled = false; continue; }
                if (p.remaining <= 0) continue;
                p.remaining -= delta;
                if (p.remaining <= 0) { p.renderer.enabled = false; continue; }
                p.velocity.y -= delta * 1.5f;
                p.position += p.velocity * delta;
                p.renderer.transform.position = new Vector3(Mathf.Round(p.position.x * 32) / 32, Mathf.Round(p.position.y * 32) / 32, 0);
            }
        }

        public void Clear() { foreach (var p in pool) { p.remaining = 0; p.renderer.enabled = false; } }
    }
}

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Tilemaps;

namespace CrystalCaves.Pilot
{
    public sealed class CaveWorld
    {
        readonly Dictionary<string, Sprite> sprites = new Dictionary<string, Sprite>();
        readonly Dictionary<string, SpriteRenderer> objects = new Dictionary<string, SpriteRenderer>();
        readonly Dictionary<string, SpriteRenderer> glows = new Dictionary<string, SpriteRenderer>();
        readonly Dictionary<string, SpriteRenderer> effects = new Dictionary<string, SpriteRenderer>();
        readonly Dictionary<string, SpriteRenderer> shadows = new Dictionary<string, SpriteRenderer>();
        readonly HashSet<string> seen = new HashSet<string>();
        readonly List<Object> terrainAssets = new List<Object>();
        readonly Transform root;
        readonly Tilemap terrain;
        readonly Sprite glowSprite;
        readonly Material glowMaterial;
        readonly Material stoneMaterial;
        readonly Sprite shadowSprite;
        readonly CaveAtmosphere atmosphere;
        readonly CaveParticles particles;
        readonly Camera camera;
        readonly CaveHazards hazards;
        readonly SpriteRenderer[] dust = new SpriteRenderer[34];
        readonly SpriteRenderer lamp;
        int level = -2, episode = -1;
        bool oldGrounded;
        float landingUntil;
        float shake;
        int oldHealth = 3;
        // Render interpolation: the simulation steps at 60 Hz while the display may
        // refresh far faster. Each frame draws previous → current by Alpha.
        readonly Dictionary<string, Vector2> previousPositions = new Dictionary<string, Vector2>();
        readonly Dictionary<string, Vector2> currentPositions = new Dictionary<string, Vector2>();
        Vector2 previousPlayer, currentPlayer;
        const float TeleportPixels = 48;
        /// <summary>0 shows the previous simulation step, 1 the latest. Fixtures and pauses use 1.</summary>
        public float Alpha { get; set; } = 1;
        /// <summary>World units per snapped pixel: authored pixels, or display pixels for smooth motion.</summary>
        static float snapUnit = 32;

        public CaveWorld(Camera camera)
        {
            this.camera = camera;
            root = new GameObject("Cave · presentation only").transform;
            var grid = new GameObject("Authored tile grid", typeof(Grid));
            grid.transform.SetParent(root);
            var tiles = new GameObject("Terrain", typeof(Tilemap), typeof(TilemapRenderer));
            tiles.transform.SetParent(grid.transform, false);
            terrain = tiles.GetComponent<Tilemap>();
            tiles.GetComponent<TilemapRenderer>().sortingOrder = 0;
            stoneMaterial = new Material(Resources.Load<Shader>("CaveStone"));
            tiles.GetComponent<TilemapRenderer>().sharedMaterial = stoneMaterial;
            glowSprite = MakeGlow();
            glowMaterial = new Material(Resources.Load<Shader>("CaveGlow"));
            shadowSprite = MakeShadow();
            atmosphere = new CaveAtmosphere(camera, root, glowSprite, glowMaterial, stoneMaterial);
            particles = new CaveParticles(root, glowSprite);
            hazards = new CaveHazards(root, particles);
            lamp = NewSprite("Helmet light", root, -1);
            lamp.sprite = glowSprite;
            lamp.sharedMaterial = glowMaterial;
            lamp.enabled = false;
            lamp.transform.localScale = Vector3.one * 4;
            for (var index = 0; index < dust.Length; index++)
            {
                dust[index] = NewSprite("Suspended dust", root, 5);
                dust[index].sprite = Pixel();
                dust[index].enabled = false;
                dust[index].transform.localScale = Vector3.one * .035f;
            }
        }

        public void Apply(CaveSnapshot state)
        {
            if (state.player == null) return;
            var newEpisode = episode != state.episode;
            if (newEpisode)
            {
                episode = state.episode; landingUntil = 0; oldGrounded = state.player.grounded;
                foreach (var effect in effects.Values) Object.Destroy(effect.gameObject); effects.Clear();
                particles.Clear(); oldHealth = state.health;
            }
            if (!newEpisode && !oldGrounded && state.player.grounded && state.steps > 0)
            { landingUntil = CaveVisualClock.Now + .13f; particles.Burst(state.player.x + 11, PlayerY(state, 29), "land"); }
            oldGrounded = state.player.grounded;
            var levelChanged = level != state.level;
            if (levelChanged)
            {
                level = state.level;
                BuildTerrain(state);
                atmosphere.Build(state);
                hazards.Build(state);
                particles.Clear();
                oldHealth = state.health;
                camera.transform.position = CameraTarget(state);
            }
            if (state.health < oldHealth)
            { shake = .16f; particles.Burst(state.player.x + 12, state.player.y + 16, "damage"); }
            oldHealth = state.health;
            RecordPositions(state, newEpisode || levelChanged);
            seen.Clear();
            // Entity IDs are reused across rooms. A ground enemy can become a
            // flyer; discard its previous contact shadow before applying this frame.
            foreach (var shadow in shadows.Values) shadow.gameObject.SetActive(false);
            foreach (var entity in state.entities)
            {
                seen.Add(entity.id);
                if (!objects.TryGetValue(entity.id, out var renderer))
                {
                    renderer = NewSprite(entity.id, root, entity.id == "player" ? 4 : 3);
                    if (entity.id == "player" || entity.id.StartsWith("enemy_")) renderer.sharedMaterial = stoneMaterial;
                    objects.Add(entity.id, renderer);
                }
                renderer.gameObject.SetActive(true);
                renderer.sprite = LoadSprite(AnimatedSprite(entity, state));
                renderer.flipX = entity.flip;
                renderer.flipY = entity.id == "player" && GravitySign(state) < 0;
                renderer.transform.position = Snap(EntityPosition(entity, renderer.sprite));
                if (entity.id == "player")
                    renderer.color = Color.white;
                if (!string.IsNullOrEmpty(entity.glow))
                {
                    if (!glows.TryGetValue(entity.id, out var glow))
                    {
                        glow = NewSprite(entity.id + " · light", root, 1);
                        glow.sprite = glowSprite;
                        glow.sharedMaterial = glowMaterial;
                        glow.transform.localScale = Vector3.one * 2.2f;
                        glows.Add(entity.id, glow);
                    }
                    glow.gameObject.SetActive(false);
                    glow.transform.position = renderer.transform.position;
                    glow.color = entity.glow == "amber" ? new Color(1, .55f, .16f, .27f)
                        : entity.glow == "violet" ? new Color(.66f, .3f, 1, .24f) : new Color(.15f, .68f, 1, .27f);
                }
                if (entity.id == "player" || entity.sprite == "slug_enemy" || entity.sprite == "dinosaur_enemy" || entity.sprite == "walking_rock")
                {
                    if (!shadows.TryGetValue(entity.id, out var shadow))
                    {
                        shadow = NewSprite(entity.id + " · contact shadow", root, 2);
                        shadow.sprite = shadowSprite; shadow.color = new Color(.015f, .025f, .03f, .45f);
                        shadows.Add(entity.id, shadow);
                    }
                    shadow.gameObject.SetActive(entity.id != "player" || state.player.grounded);
                    var supportY = entity.id == "player" && GravitySign(state) < 0
                        ? entity.y + 1 : entity.y + entity.h - 1;
                    shadow.transform.position = Snap(Position(entity.x + entity.w / 2, supportY));
                }
            }
            foreach (var pair in objects) if (!seen.Contains(pair.Key)) pair.Value.gameObject.SetActive(false);
            foreach (var pair in glows) if (!seen.Contains(pair.Key)) pair.Value.gameObject.SetActive(false);
            foreach (var pair in shadows) if (!seen.Contains(pair.Key)) pair.Value.gameObject.SetActive(false);
            lamp.transform.position = Position(state.player.x + 11, PlayerY(state, 10));
            UpdateEffects(state.effects);
        }

        void UpdateEffects(CaveEffect[] events)
        {
            seen.Clear();
            if (events != null) foreach (var effect in events)
            {
                seen.Add(effect.id);
                if (!effects.TryGetValue(effect.id, out var renderer))
                {
                    renderer = NewSprite("Feedback · " + effect.kind, root, 6);
                    if (effect.kind != "bones" && effect.kind != "slime_pulse" && effect.text != "OUCH")
                        particles.Burst(effect.x, effect.y, effect.text != null && effect.text.StartsWith("-") ? "damage" : effect.kind);
                    // Damage already has a white actor pose and its label.
                    // The old gold sparkle obscured those authored pixels.
                    renderer.enabled = effect.kind != "score" && effect.text != "OUCH";
                    if (effect.kind != "bones" && effect.kind != "slime_pulse") renderer.sprite = LoadSprite(effect.kind == "poof" ? "poof" : "sparkle");
                    effects.Add(effect.id, renderer);
                }
                if (effect.kind == "bones")
                {
                    var age = Mathf.Max(0, effect.max_ttl - effect.ttl);
                    var phase = Mathf.Min(3, age * 4 / Mathf.Max(1, effect.max_ttl));
                    var direction = effect.facing < 0 ? -1 : 1;
                    renderer.sprite = LoadSprite("defeat_bones_" + phase);
                    renderer.flipX = direction < 0;
                    renderer.color = Color.white;
                    renderer.transform.position = Snap(Position(effect.x + phase * 4 * direction, effect.y + phase * (phase + 1) / 2));
                    renderer.transform.localScale = Vector3.one;
                    renderer.transform.rotation = Quaternion.identity;
                    continue;
                }
                if (effect.kind == "slime_pulse")
                {
                    var age = Mathf.Max(0, effect.max_ttl - effect.ttl);
                    var phase = Mathf.Min(3, age * 4 / Mathf.Max(1, effect.max_ttl));
                    var direction = effect.facing < 0 ? -1 : 1;
                    renderer.sprite = LoadSprite("defeat_slime_" + phase);
                    renderer.flipX = direction < 0;
                    renderer.color = Color.white;
                    renderer.transform.position = Snap(Position(effect.x + phase * 5 * direction, effect.y));
                    renderer.transform.localScale = Vector3.one;
                    renderer.transform.rotation = Quaternion.identity;
                    continue;
                }
                var life = effect.ttl / (float)Mathf.Max(1, effect.max_ttl);
                renderer.color = new Color(1, 1, 1, life);
                renderer.transform.position = Position(effect.x, effect.y);
                renderer.transform.localScale = Vector3.one * .5f;
                renderer.transform.rotation = Quaternion.identity;
            }
            var expired = new List<string>();
            foreach (var pair in effects) if (!seen.Contains(pair.Key)) expired.Add(pair.Key);
            foreach (var key in expired) { Object.Destroy(effects[key].gameObject); effects.Remove(key); }
        }

        public void Animate(CaveSnapshot state, float delta)
        {
            if (state?.player == null) return;
            // Every authored pixel covers an integer number of display pixels.
            camera.pixelRect = CaveViewport.CameraRect;
            camera.orthographicSize = camera.pixelHeight / (64f * CaveViewport.Scale);
            snapUnit = CaveSettings.Data.motion == 2 ? 32f * CaveViewport.Scale : 32f;
            var playerPosition = Vector2.LerpUnclamped(previousPlayer, currentPlayer, Alpha);
            var smooth = CameraTarget(state, playerPosition.x, playerPosition.y);
            shake = Mathf.Max(0, shake - delta);
            if (shake > 0 && CaveVisualSettings.Motion && CaveSettings.Data.shake > 0) smooth += new Vector3(Mathf.Sin(CaveVisualClock.Now * 83), Mathf.Cos(CaveVisualClock.Now * 71), 0) * shake * .35f * CaveSettings.Data.shake / 100f;
            camera.transform.position = Snap(smooth);
            atmosphere.Animate(state);
            hazards.Animate(camera, delta);
            particles.Animate(Mathf.Min(delta, .05f));
            if (state.effects != null) foreach (var effect in state.effects)
            {
                if (!effects.TryGetValue(effect.id, out var renderer)) continue;
                var essential = effect.kind == "bones" || effect.kind == "slime_pulse";
                renderer.enabled = essential || CaveVisualSettings.Motion && CaveSettings.Data.particles && CaveSettings.Data.effectDensity > 0 && effect.kind != "score" && effect.text != "OUCH";
            }
            foreach (var entity in state.entities)
            {
                if (!objects.TryGetValue(entity.id, out var renderer)) continue;
                renderer.sprite = LoadSprite(AnimatedSprite(entity, state));
                var blended = Blend(entity);
                var position = EntityPosition(entity, renderer.sprite, blended.x, blended.y);
                renderer.transform.position = Snap(position);
                if (glows.TryGetValue(entity.id, out var glow))
                {
                    glow.gameObject.SetActive(renderer.gameObject.activeSelf && CaveSettings.Data.lighting > 0);
                    glow.transform.position = renderer.transform.position;
                    var tint = glow.color; tint.a = CaveSettings.Data.lighting / 100f * .35f; glow.color = tint;
                }
                if (shadows.TryGetValue(entity.id, out var shadow))
                {
                    var groundedActor = entity.id == "player" ? state.player.grounded : entity.sprite == "slug_enemy" || entity.sprite == "dinosaur_enemy" || entity.sprite == "walking_rock";
                    shadow.gameObject.SetActive(groundedActor && renderer.gameObject.activeSelf && CaveSettings.Data.contactShadows);
                    shadow.color = new Color(.015f,.025f,.03f,CaveSettings.Data.shadowStrength / 100f);
                    var supportY = entity.id == "player" && GravitySign(state) < 0 ? blended.y + 1 : blended.y + entity.h - 1;
                    shadow.transform.position = Snap(Position(blended.x + entity.w / 2, supportY));
                }
            }
            lamp.transform.position = Position(playerPosition.x + 11, PlayerY(state, playerPosition.y, 10));
            for (var index = 0; index < dust.Length; index++)
            {
                var x = Mathf.Repeat(index * 7.13f + CaveVisualClock.Now * .12f, 30) - 15;
                var y = Mathf.Repeat(index * 3.47f + CaveVisualClock.Now * .08f, 18) - 9;
                dust[index].enabled = CaveVisualSettings.Motion && CaveSettings.Data.environment && CaveSettings.Data.particles && index < (CaveSettings.Data.effectDensity == 2 ? 12 : CaveSettings.Data.effectDensity == 1 ? 4 : 0);
                dust[index].color = new Color(.66f,.66f,.66f,.2f);
                dust[index].transform.localScale = Vector3.one / 32;
                dust[index].transform.position = Snap(new Vector3(smooth.x + x, smooth.y + y, 0));
            }
            foreach (var pair in glows)
            {
                if (!pair.Value.gameObject.activeSelf) continue;
                var pulse = 1.65f + (CaveVisualSettings.Motion && CaveSettings.Data.environment ? Mathf.Sin(CaveVisualClock.Now * 2.5f + pair.Value.transform.position.x) * .08f : 0);
                pair.Value.transform.localScale = Vector3.one * pulse;
            }
        }

        string AnimatedSprite(CaveEntity entity, CaveSnapshot state)
        {
            if (entity.id == "player")
            {
                var pose = PlayerSprite(entity, state);
                // Use the original immunity clock, beginning with a white phase.
                // Older recorded protocol snapshots fall back to their step clock.
                var age = state.player.invulnerability_frames > 0
                    ? Mathf.Max(0, state.player.invulnerability_frames - state.player.invulnerability_left) : state.steps;
                return state.player.invulnerable && (!CaveVisualSettings.Motion || !CaveSettings.Data.damageFlashes || age / 12 % 2 == 0) ? pose + "_hit" : pose;
            }
            if (entity.id.StartsWith("torch_")) return "mine_torch_" + (CaveVisualSettings.Motion ? state.steps / 8 % 4 : 0);
            if (entity.id.StartsWith("lift_")) return "elevator_" + (entity.frame);
            if (entity.id.StartsWith("bullet_")) return "bullet_" + (state.steps / 3 % 4);
            if (entity.sprite == "air_tank_tall")
                return "air_tank_tall_" + (state.steps / 7 % 2);
            if (entity.asleep) return "walking_rock_sleep";
            if (entity.id.StartsWith("enemy_"))
            {
                var pose = (state.steps + state.freeze_timer) / 6 % 4;
                var hit = entity.hit && entity.sprite == "dinosaur_enemy" ? "_hit" : "";
                return entity.sprite + hit + "_" + pose;
            }
            if (entity.id.StartsWith("crystal_"))
                {
                var index = (Mathf.RoundToInt(entity.x / 32) + Mathf.RoundToInt(entity.y / 32) * 3) % 4;
                var name = new[] { "crystal_blue", "crystal_green", "crystal_yellow", "crystal_red" }[index];
                return CaveVisualSettings.Motion && CaveSettings.Data.gemGlints && (state.steps / 10 + index) % 13 == 0 ? name + "_glint" : name;
            }
            return entity.sprite;
        }

        string PlayerSprite(CaveEntity entity, CaveSnapshot state)
        {
            if (state.steps == 0) return "mylo_idle";
            if (entity.sprite == "mylo_shoot" || entity.sprite == "mylo_shoot_air")
                return !state.player.grounded && !state.player.climbing ? "mylo_shoot_air" : "mylo_shoot";
            if (state.player.climbing) return "mylo_climb_" + (state.steps / 8 % 2);
            if (!state.player.grounded) return state.player.vy * GravitySign(state) < 0 ? "mylo_jump" : "mylo_fall";
            if (CaveVisualClock.Now < landingUntil) return "mylo_land";
            if (Mathf.Abs(state.player.vx) > .2f) return "mylo_run_" + (state.steps / 6 % 4);
            return CaveVisualSettings.Motion && Mathf.Repeat(CaveVisualClock.Now, 5) > 4.5f ? "mylo_idle_look" : "mylo_idle";
        }

        static Vector3 Snap(Vector3 value) => new Vector3(Mathf.Round(value.x * snapUnit) / snapUnit, Mathf.Round(value.y * snapUnit) / snapUnit, value.z);

        void RecordPositions(CaveSnapshot state, bool discontinuous)
        {
            var player = new Vector2(state.player.x, state.player.y);
            previousPlayer = discontinuous || Teleported(currentPlayer, player) ? player : currentPlayer;
            currentPlayer = player;
            previousPositions.Clear();
            foreach (var entity in state.entities)
            {
                var now = new Vector2(entity.x, entity.y);
                previousPositions[entity.id] = !discontinuous && currentPositions.TryGetValue(entity.id, out var before) && !Teleported(before, now) ? before : now;
            }
            currentPositions.Clear();
            foreach (var entity in state.entities) currentPositions[entity.id] = new Vector2(entity.x, entity.y);
        }

        // Respawns, doors, secret reveals and reused IDs must cut, not smear across the room.
        static bool Teleported(Vector2 from, Vector2 to) => Mathf.Abs(to.x - from.x) > TeleportPixels || Mathf.Abs(to.y - from.y) > TeleportPixels;

        Vector2 Blend(CaveEntity entity)
        {
            var now = new Vector2(entity.x, entity.y);
            if (Alpha >= 1 || !previousPositions.TryGetValue(entity.id, out var before)
                || !currentPositions.TryGetValue(entity.id, out var recorded) || recorded != now) return now;
            return Vector2.LerpUnclamped(before, now, Alpha);
        }

        // Missing gravity metadata in older clients defaults to the normal pose.
        // Mylo's sprite spans player.y - 2 through player.y + 30, so mirror
        // local presentation points around its center at player.y + 14.
        static int GravitySign(CaveSnapshot state) => state.player.gravity_dir < 0 ? -1 : 1;
        static float PlayerY(CaveSnapshot state, float normalOffset) => PlayerY(state, state.player.y, normalOffset);
        static float PlayerY(CaveSnapshot state, float playerY, float normalOffset) =>
            playerY + 14 + (normalOffset - 14) * GravitySign(state);

        // Tall art shares the authoritative patrol's floor anchor. The snake
        // keeps its 24px collision body while its upright artwork spans 32px.
        static Vector3 EntityPosition(CaveEntity entity, Sprite sprite) => EntityPosition(entity, sprite, entity.x, entity.y);
        static Vector3 EntityPosition(CaveEntity entity, Sprite sprite, float x, float y) => Position(x + entity.w / 2,
            entity.sprite == "dinosaur_enemy" || entity.sprite == "slug_enemy"
                ? y + entity.h - sprite.bounds.size.y * 16 : y + entity.h / 2);

        Vector3 CameraTarget(CaveSnapshot state) => CameraTarget(state, state.player.x, state.player.y);
        Vector3 CameraTarget(CaveSnapshot state, float playerX, float playerY)
        {
            var halfWidth = camera.orthographicSize * camera.aspect;
            return new Vector3(
                ClampCenter((playerX + 11) / 32f, halfWidth, state.cols),
                -ClampCenter((playerY + 15) / 32f, camera.orthographicSize, state.rows), -10);
        }

        static float ClampCenter(float value, float halfSize, float size) => size <= halfSize * 2 ? size / 2 : Mathf.Clamp(value, halfSize, size - halfSize);
        public static Vector3 Position(float x, float y) => new Vector3(x / 32f, -y / 32f, 0);

        void BuildTerrain(CaveSnapshot state)
        {
            terrain.ClearAllTiles();
            foreach (var asset in terrainAssets) Object.Destroy(asset);
            terrainAssets.Clear();
            var atlas = Resources.Load<Texture2D>("Terrain/level_" + (state.realm == "mine" ? "mine" : state.level.ToString()));
            for (var row = 0; row < state.rows; row++)
            for (var col = 0; col < state.cols; col++)
            {
                var symbol = state.layout[row][col];
                if (symbol != '#' && symbol != 'H' && symbol != '^' && symbol != '~') continue;
                var tile = ScriptableObject.CreateInstance<Tile>();
                tile.sprite = Sprite.Create(atlas, new Rect(col * 32, atlas.height - (row + 1) * 32, 32, 32), Vector2.one * .5f, 32);
                terrain.SetTile(new Vector3Int(col, -row - 1, 0), tile);
                terrainAssets.Add(tile.sprite);
                terrainAssets.Add(tile);
            }
        }

        /// <summary>HD creature textures have this many texels per native pixel.</summary>
        public const int HdScale = 4;

        // HD creature art replaces a pixel sprite of the same name at the same world size.
        Sprite LoadSprite(string name)
        {
            var hd = CaveSettings.Data.hdCreatures;
            var key = hd ? "hd:" + name : name;
            if (sprites.TryGetValue(key, out var sprite)) return sprite;
            var texture = hd ? Resources.Load<Texture2D>("SpritesHD/" + name) : null;
            var texelsPerPixel = texture != null ? HdScale : 1;
            if (texture == null) texture = Resources.Load<Texture2D>("Sprites/" + name);
            if (texture == null) { Debug.LogError("Missing cave sprite: " + name); return null; }
            sprite = Sprite.Create(texture, new Rect(0, 0, texture.width, texture.height), Vector2.one * .5f, 32 * texelsPerPixel);
            sprites.Add(key, sprite);
            return sprite;
        }

        static SpriteRenderer NewSprite(string name, Transform parent, int order)
        {
            var instance = new GameObject(name, typeof(SpriteRenderer));
            instance.transform.SetParent(parent, false);
            var renderer = instance.GetComponent<SpriteRenderer>();
            renderer.sortingOrder = order;
            return renderer;
        }

        static Sprite Pixel()
        {
            var texture = new Texture2D(1, 1);
            texture.SetPixel(0, 0, Color.white);
            texture.Apply();
            return Sprite.Create(texture, new Rect(0, 0, 1, 1), Vector2.one * .5f, 1);
        }

        static Sprite MakeGlow()
        {
            var texture = new Texture2D(64, 64, TextureFormat.RGBA32, false) { filterMode = FilterMode.Bilinear };
            for (var y = 0; y < 64; y++) for (var x = 0; x < 64; x++)
            {
                var radius = Vector2.Distance(new Vector2(x, y), new Vector2(31.5f, 31.5f)) / 32;
                texture.SetPixel(x, y, new Color(1, 1, 1, Mathf.Pow(Mathf.Max(0, 1 - radius), 2.8f)));
            }
            texture.Apply();
            return Sprite.Create(texture, new Rect(0, 0, 64, 64), Vector2.one * .5f, 64);
        }

        static Sprite MakeShadow()
        {
            var texture = new Texture2D(24, 6, TextureFormat.RGBA32, false) { filterMode = FilterMode.Point };
            for (var y = 0; y < 6; y++) for (var x = 0; x < 24; x++)
            {
                var dx = (x - 11.5f) / 12; var dy = (y - 2.5f) / 3;
                texture.SetPixel(x, y, dx * dx + dy * dy < 1 ? Color.white : Color.clear);
            }
            texture.Apply();
            return Sprite.Create(texture, new Rect(0, 0, 24, 6), Vector2.one * .5f, 32);
        }
    }
}

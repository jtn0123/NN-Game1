using System;
using System.Collections.Generic;
using CrystalCaves.Sim;

namespace CrystalCaves.Pilot
{
    /// <summary>Where authoritative cave state comes from. At most one command is in flight.</summary>
    public interface ICaveBackend : IDisposable
    {
        bool Connected { get; }
        bool Busy { get; }
        /// <summary>True when the simulation runs inside this process.</summary>
        bool InProcess { get; }
        bool Send(CaveCommand command);
        /// <summary>Returns queued replies in order. A null snapshot means a malformed reply.</summary>
        bool Receive(out CaveSnapshot snapshot);
    }

    /// <summary>The optional Python bridge: AI checkpoints, demo recording and native smoke checks.</summary>
    public sealed class BridgeBackend : ICaveBackend
    {
        readonly CaveConnection connection;
        public BridgeBackend(int port) { connection = new CaveConnection(port); }
        public bool Connected => connection.Connected;
        public bool Busy => connection.Busy;
        public bool InProcess => false;
        public bool Send(CaveCommand command) => connection.Send(UnityEngine.JsonUtility.ToJson(command));
        public bool Receive(out CaveSnapshot snapshot)
        {
            snapshot = null;
            if (!connection.Read(out var json)) return false;
            if (!CaveSnapshotReader.TryRead(json, out snapshot)) snapshot = null;
            return true;
        }
        public void Dispose() => connection.Dispose();
    }

    /// <summary>
    /// The standalone game: the C# simulation, verified frame by frame against the
    /// Python reference by <c>unity/SimTests</c>, stepped synchronously on the main thread.
    /// </summary>
    public sealed class LocalCaveBackend : ICaveBackend
    {
        public const string AiUnavailable = "Watching the AI needs its Python checkpoint. Start the game with scripts/unity_pilot.py --model <checkpoint>.";
        static readonly string[] ActionLabels = { "IDLE", "LEFT", "RIGHT", "JUMP", "LEFT_JUMP", "RIGHT_JUMP", "SHOOT", "LEFT_SHOOT", "RIGHT_SHOOT", "INTERACT" };

        readonly CaveSim sim = new CaveSim();
        readonly Queue<CaveSnapshot> replies = new Queue<CaveSnapshot>();
        readonly List<HumanControls> controls = new List<HumanControls>(CaveSim.MaxBatch);

        public bool Connected => true;
        public bool Busy => false;
        public bool InProcess => true;

        public bool Send(CaveCommand command)
        {
            replies.Enqueue(Execute(command));
            return true;
        }

        public bool Receive(out CaveSnapshot snapshot)
        {
            if (replies.Count == 0) { snapshot = null; return false; }
            snapshot = replies.Dequeue();
            return true;
        }

        public void Dispose() => replies.Clear();

        CaveSnapshot Execute(CaveCommand command)
        {
            try
            {
                switch (command?.op ?? "snapshot")
                {
                    case "snapshot": return Convert(sim.Snapshot());
                    case "reset": return Convert(sim.Reset(command.level));
                    case "mine": return Convert(sim.OpenMine(command.cleared ?? Array.Empty<int>()));
                    case "mode":
                        if (command.mode == "human") return Convert(sim.Snapshot());
                        return Error(command.mode == "ai" ? AiUnavailable : "Mode must be human or ai.");
                    case "human_step":
                        if (command.controls == null) return Error("Controls are required.");
                        controls.Clear();
                        foreach (var control in command.controls)
                        {
                            if (control == null) return Error("Each control needs move, jump, shoot and interact.");
                            controls.Add(new HumanControls(control.move, control.jump, control.shoot, control.interact));
                        }
                        return Convert(sim.HumanStep(controls));
                    case "step": return Error("Numbered actions belong to the AI bridge. Human play sends independent buttons.");
                    default: return Error("Unknown command " + command.op + ".");
                }
            }
            catch (ArgumentException error) { return Error(error.Message); }
            catch (InvalidOperationException error) { return Error(error.Message); }
        }

        static CaveSnapshot Error(string message) => new CaveSnapshot { protocol = 1, error = message };

        static CaveSnapshot Convert(SimSnapshot source)
        {
            var player = source.Player;
            var entities = new CaveEntity[source.Entities.Count];
            for (var index = 0; index < entities.Length; index++)
            {
                var entity = source.Entities[index];
                entities[index] = new CaveEntity
                {
                    id = entity.Id, sprite = entity.Sprite, glow = entity.Glow,
                    x = (float)entity.X, y = (float)entity.Y, w = (float)entity.W, h = (float)entity.H,
                    flip = entity.Flip, asleep = entity.Asleep, hit = entity.Hit, frame = entity.Frame,
                };
            }
            var effects = new CaveEffect[source.Effects.Count];
            for (var index = 0; index < effects.Length; index++)
            {
                var effect = source.Effects[index];
                effects[index] = new CaveEffect
                {
                    id = effect.Id, kind = effect.Kind, text = effect.Text,
                    x = (float)effect.X, y = (float)effect.Y, ttl = effect.Ttl, max_ttl = effect.MaxTtl, facing = effect.Facing,
                };
            }
            return new CaveSnapshot
            {
                protocol = 1, realm = source.Realm, cleared_caves = source.ClearedCaves,
                near_entrance = source.NearEntrance, portal_level = source.PortalLevel,
                episode = source.Episode, level = source.Level, level_name = source.LevelName,
                levels = ToArray(source.Levels), layout = ToArray(source.Layout),
                cols = source.Cols, rows = source.Rows,
                player = new CavePlayer
                {
                    x = (float)player.X, y = (float)player.Y, vx = (float)player.Vx, vy = (float)player.Vy,
                    sprite = player.Sprite, facing = player.Facing, gravity_dir = player.GravityDir,
                    invulnerability_left = player.InvulnerabilityLeft, invulnerability_frames = player.InvulnerabilityFrames,
                    grounded = player.Grounded, invulnerable = player.Invulnerable, climbing = player.Climbing,
                },
                entities = entities, effects = effects,
                health = source.Health, ammo = source.Ammo, score = source.Score,
                crystals = source.Crystals, initial_crystals = source.InitialCrystals,
                steps = source.Steps, freeze_timer = source.FreezeTimer, super_timer = source.SuperTimer,
                exit_unlocked = source.ExitUnlocked, done = source.Done, won = source.Won,
                end_reason = source.EndReason, mode = source.Mode, human_only = source.HumanOnly,
                sounds = ToArray(source.Sounds),
                policy_name = "", action_labels = ActionLabels, q_values = Array.Empty<float>(),
            };
        }

        static string[] ToArray(IReadOnlyList<string> values)
        {
            var result = new string[values.Count];
            for (var index = 0; index < result.Length; index++) result[index] = values[index];
            return result;
        }
    }
}

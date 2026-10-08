using System;

namespace CrystalCaves.Pilot
{
    [Serializable] public sealed class CaveCommand
    {
        public string op;
        public int[] actions, cleared;
        public CaveHumanControl[] controls;
        public int level;
        public string mode;
    }

    [Serializable] public sealed class CaveHumanControl
    {
        public int move;
        public bool jump, shoot, interact;
    }

    [Serializable] public sealed class CavePlayer
    {
        public float x, y, vx, vy;
        public string sprite;
        public int facing, gravity_dir, invulnerability_left, invulnerability_frames;
        public bool grounded, invulnerable, climbing;
    }

    [Serializable] public sealed class CaveEntity
    {
        public string id, sprite, glow;
        public float x, y, w, h;
        public int frame;
        public bool flip, asleep, hit;
    }

    [Serializable] public sealed class CaveEffect
    {
        public string id, kind, text;
        public float x, y;
        public int ttl, max_ttl, facing;
    }

    [Serializable] public sealed class CaveSnapshot
    {
        public int protocol, episode, level, cols, rows;
        public string error, level_name, mode, policy_name, end_reason, realm;
        public int near_entrance, portal_level, cleared_caves;
        public string[] levels, layout, sounds, action_labels;
        public CavePlayer player;
        public CaveEntity[] entities;
        public CaveEffect[] effects;
        public int health, ammo, score, crystals, initial_crystals, steps, max_steps, freeze_timer, super_timer;
        public int stall_steps, stall_limit, state_size, action, demos_saved;
        public bool exit_unlocked, done, won, ai_available, recording, human_only, training_limits;
        public float last_reward, total_reward;
        public float[] q_values;
    }

    [Serializable] public sealed class CaveInfo
    {
        public string name, region;
        public int crystals, cols, rows, theme;
        public CaveLightInfo[] lights;
    }
    [Serializable] public sealed class CaveLightInfo { public float x, y; }
    [Serializable] public sealed class CaveCatalog { public CaveInfo[] caves; }
}

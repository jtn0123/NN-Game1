using System.Collections.Generic;
using System;
using UnityEngine;

namespace CrystalCaves.Pilot
{
    public sealed class CaveAudio
    {
        [Serializable] sealed class SoundEntry { public string name; public int priority; }
        [Serializable] sealed class SoundCatalog { public SoundEntry[] sounds; }
        readonly AudioSource effects;
        readonly Dictionary<string, AudioClip> clips = new Dictionary<string, AudioClip>();
        readonly Dictionary<string, int> priorities = new Dictionary<string, int>();
        int currentPriority;
        bool playing = true;
        float previewUntil;
        public bool Muted { get; private set; }
        public float Volume { get; private set; } = .7f;
        public string CurrentSound => effects.clip != null ? effects.clip.name : "";
        public bool IsPlaying => effects.isPlaying;

        public CaveAudio(GameObject owner)
        {
            effects = owner.AddComponent<AudioSource>();
            effects.playOnAwake = false;
            effects.volume = .5f;
            var catalog = Resources.Load<TextAsset>("ClassicAudio");
            if (catalog != null)
                foreach (var sound in JsonUtility.FromJson<SoundCatalog>(catalog.text).sounds)
                    priorities[sound.name] = sound.priority;
            Refresh();
            CaveSettings.Changed += Refresh;
            owner.AddComponent<CaveAudioCleanup>().Cleanup = () => CaveSettings.Changed -= Refresh;
        }

        public void Play(string[] events)
        {
            if (events == null) return;
            foreach (var name in events)
            {
                if (name == "land") continue;
                if (!clips.TryGetValue(name, out var clip))
                {
                    clip = Resources.Load<AudioClip>("Audio/" + name);
                    clips[name] = clip;
                }
                if (clip == null) continue;
                var priority = priorities.TryGetValue(name, out var value) ? value : 0;
                // One speaker: lower-priority events cannot interrupt a cue.
                // Equal or higher priority takes over immediately, as in the classic.
                if (effects.isPlaying && priority < currentPriority) continue;
                currentPriority = priority;
                effects.clip = clip;
                effects.Play();
                if (!playing) effects.Pause();
            }
        }

        public void SetPlaying(bool active)
        {
            if (Time.unscaledTime < previewUntil) active = true;
            if (playing == active) return;
            playing = active;
            if (active) effects.UnPause();
            else effects.Pause();
        }

        public void Preview(string name)
        {
            previewUntil = Time.unscaledTime + 2;
            SetPlaying(true); effects.Stop(); currentPriority = 0;
            Play(new[] { name });
        }
        public void ToggleMute() => CaveSettings.Change(d => d.muted = !d.muted, false);
        public void SetVolume(float volume) => CaveSettings.Change(d => d.masterVolume = Mathf.RoundToInt(Mathf.Clamp01(volume) * 100), false);
        void Refresh()
        {
            Volume = CaveSettings.Data.masterVolume / 100f;
            Muted = CaveSettings.Data.muted;
            effects.volume = Volume * CaveSettings.Data.effectsVolume / 100f * .65f;
            effects.mute = Muted;
        }
    }
    public sealed class CaveAudioCleanup : MonoBehaviour
    {
        public Action Cleanup;
        void OnDestroy() { Cleanup?.Invoke(); }
    }
}

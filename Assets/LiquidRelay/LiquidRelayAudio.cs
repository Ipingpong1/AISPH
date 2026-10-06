using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>Quiet procedural feedback; the demo needs no downloaded audio assets.</summary>
public sealed class LiquidRelayAudio : MonoBehaviour
{
    public LiquidRelayGame game;
    [Range(0, 1)] public float volume = .16f;
    public bool soundEnabled = true;
    AudioSource source;
    AudioClip release, success, retry, reset;
    LiquidRelayGame.RoundState previous;
    int previousRound;
    bool initialized;

    void Start()
    {
        if (game == null) game = GetComponent<LiquidRelayGame>();
        source = gameObject.AddComponent<AudioSource>();
        source.playOnAwake = false; source.spatialBlend = 0; source.volume = volume;
        release = Chime("Relay release", new[] { 220f, 330f }, .10f, .38f);
        success = Chime("Relay complete", new[] { 523.25f, 659.25f, 783.99f, 1046.5f }, .11f, .48f);
        retry = Chime("Relay retry", new[] { 329.63f, 261.63f }, .17f, .40f);
        reset = Chime("Relay reset", new[] { 440f }, .08f, .16f);
        if (game != null) { previous = game.State; previousRound = game.RoundNumber; initialized = true; }
    }

    void Update()
    {
        if (Keyboard.current != null && Keyboard.current.mKey.wasPressedThisFrame) ToggleSound();
        if (!initialized || game == null) return;
        source.volume = volume;
        if (game.RoundNumber != previousRound)
        {
            Play(reset); previousRound = game.RoundNumber;
        }
        else if (game.State != previous)
        {
            if (game.State == LiquidRelayGame.RoundState.Running) Play(release);
            else if (game.State == LiquidRelayGame.RoundState.Won) Play(success);
            else if (game.State == LiquidRelayGame.RoundState.Lost) Play(retry);
        }
        previous = game.State;
    }

    public void ToggleSound()
    {
        soundEnabled = !soundEnabled;
        if (!soundEnabled && source != null) source.Stop();
    }

    void Play(AudioClip clip) { if (soundEnabled && source != null) source.PlayOneShot(clip); }

    static AudioClip Chime(string title, float[] notes, float spacing, float decay)
    {
        const int rate = 24000;
        int length = Mathf.CeilToInt(((notes.Length - 1) * spacing + decay * 3) * rate);
        var samples = new float[length];
        for (int n = 0; n < notes.Length; n++)
        {
            int start = Mathf.RoundToInt(n * spacing * rate);
            for (int i = start; i < length; i++)
            {
                float t = (float)(i - start) / rate;
                float envelope = Mathf.Min(1, t / .008f) * Mathf.Exp(-t / decay * 3.5f);
                float fundamental = Mathf.Sin(2 * Mathf.PI * notes[n] * t);
                float overtone = .22f * Mathf.Sin(2 * Mathf.PI * notes[n] * 2 * t) * Mathf.Exp(-t * 14);
                samples[i] += (fundamental + overtone) * envelope * .26f;
            }
        }
        var clip = AudioClip.Create(title, length, 1, rate, false);
        clip.SetData(samples, 0);
        return clip;
    }

    void OnDestroy()
    {
        if (release != null) Destroy(release);
        if (success != null) Destroy(success);
        if (retry != null) Destroy(retry);
        if (reset != null) Destroy(reset);
    }
}

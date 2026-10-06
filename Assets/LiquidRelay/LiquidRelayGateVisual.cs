using UnityEngine;

/// <summary>The collision gate lifts at full size; its visible shutter retracts into its top housing.</summary>
[DefaultExecutionOrder(100)]
public sealed class LiquidRelayGateVisual : MonoBehaviour
{
    public LiquidRelayGame game;
    public Transform shutter;
    public float gateHeight = 1.54f;
    public float liftDistance = 1.62f;
    Renderer shutterRenderer;

    void Start() { if (shutter != null) shutterRenderer = shutter.GetComponent<Renderer>(); }

    void LateUpdate()
    {
        if (game == null || shutter == null) return;
        float amount = liftDistance * game.GateFraction / gateHeight;
        float remaining = Mathf.Max(0, 1 - amount);
        shutter.localPosition = new Vector3(0, -amount * .5f, 0);
        shutter.localScale = new Vector3(1, remaining, 1);
        if (shutterRenderer != null) shutterRenderer.enabled = remaining > .001f;
    }
}

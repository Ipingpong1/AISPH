// MouseStirrer.cs — the live-demo "splash" tool. Attach to an obstacle sphere that is listed in the
// provider's obstacles (motion = None). Hold LEFT mouse: the sphere follows the mouse ray on a
// horizontal plane at its current height; the scroll wheel raises/lowers it. The move is smoothed
// so the PBF obstacle projection imparts a bounded velocity to the fluid (the solver has no
// explicit force term — pushing IS the obstacle projection, which is why this needs no solver
// change). Right mouse stays the camera look (FluidSceneMVP). Position is clamped to the sim
// domain in world space (anchor at the origin, [0,3]³ centred on XZ => x,z in [-1.5,1.5]).
using UnityEngine;
using UnityEngine.InputSystem;

public class MouseStirrer : MonoBehaviour
{
    [Tooltip("Camera whose mouse ray drives the sphere. Empty = Camera.main.")]
    public Camera cam;
    [Tooltip("Exponential follow rate (1/s). Higher = snappier stirring = bigger splashes.")]
    public float followSpeed = 10f;
    [Tooltip("Height change per scroll notch (world units).")]
    public float scrollHeightStep = 0.05f;
    public float minY = 0.05f, maxY = 2.5f;
    public Vector2 xzMin = new Vector2(-1.45f, -1.45f), xzMax = new Vector2(1.45f, 1.45f);
    [Tooltip("Set by SetTarget() for scripted/MCP tests; cleared by the next mouse drag.")]
    public bool scripted;
    [Tooltip("Read the mouse (LMB drag / scroll). TouchControls turns this off while it drives the stirrer.")]
    public bool mouseInput = true;

    Vector3 target;

    void Start()
    {
        if (cam == null) cam = Camera.main;
        target = transform.position;
    }

    public void SetTarget(Vector3 worldPos)
    {
        target = new Vector3(Mathf.Clamp(worldPos.x, xzMin.x, xzMax.x), Mathf.Clamp(worldPos.y, minY, maxY),
                             Mathf.Clamp(worldPos.z, xzMin.y, xzMax.y));
        scripted = true;
    }

    /// <summary>Move the target under a screen point (pixels, bottom-left origin) on the current height plane.</summary>
    public void DriveTo(Vector2 screenPos)
    {
        if (cam == null) return;
        scripted = false;
        Ray r = cam.ScreenPointToRay(screenPos);
        var plane = new Plane(Vector3.up, new Vector3(0f, target.y, 0f));
        if (plane.Raycast(r, out float t))
        {
            Vector3 h = r.GetPoint(t);
            target.x = Mathf.Clamp(h.x, xzMin.x, xzMax.x);
            target.z = Mathf.Clamp(h.z, xzMin.y, xzMax.y);
        }
    }

    public void NudgeHeight(float dy) => target.y = Mathf.Clamp(target.y + dy, minY, maxY);

    void Update()
    {
        var ms = Mouse.current;
        if (mouseInput && ms != null && cam != null)
        {
            float scroll = ms.scroll.ReadValue().y;
            if (Mathf.Abs(scroll) > 0.01f) NudgeHeight(Mathf.Sign(scroll) * scrollHeightStep);
            if (ms.leftButton.isPressed) DriveTo(ms.position.ReadValue());
        }
        float a = 1f - Mathf.Exp(-followSpeed * Time.deltaTime);
        transform.position = Vector3.Lerp(transform.position, target, a);
    }
}

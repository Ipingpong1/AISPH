
    // ---------- 2026-09-30 overnight (PBF FT-RS, ReseedRunner.cs): arbitrary-state upload + raw relaxation substeps ----------
    // Additive only: the live scene and SimExportRunner never call these.

    /// <summary>Replace the whole particle set with the first `count` entries of pos / vel (density buffer = spawn value).</summary>
    public int SetState(Vector3[] pos, Vector3[] vel, int count)
    {
        n = Mathf.Min(count, maxParticles);
        if (n == 0) return 0;
        bPos.SetData(pos, 0, 0, n);
        bVel.SetData(vel, 0, 0, n);
        for (int i = 0; i < n; i++) spawnDensStage[i] = restDensity * 0.8f;
        bDens.SetData(spawnDensStage, 0, 0, n);
        return n;
    }

    /// <summary>Zero every active particle's velocity (relaxation: drop the overlap fix-up momentum before each substep).</summary>
    public void ZeroVelocities()
    {
        if (n == 0) return;
        Array.Clear(spawnStage, 0, n);
        bVel.SetData(spawnStage, 0, 0, n);
    }

    /// <summary>One raw substep of dt seconds (no CFL, no beforeSubstep) with the current gravity / obstacles.</summary>
    public void SubstepOnce(float dt)
    {
        if (n == 0 || dt <= 0f) return;
        cs.SetInt("_N", n);
        Substep(dt);
    }

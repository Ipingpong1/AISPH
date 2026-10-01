// IGpuSimSource.cs — DFSPH1001: what the GPU render path (GpuSplatPass), the GPU spray (GpuSprayLayer) and the parity
// tools need from a live GPU solver, so the GPU PBF (GpuSphProvider) and the GPU DFSPH (DfsphProvider) are
// interchangeable behind FluidSceneMVP.provider. All buffers are in SIM space (SPlisHSPlasH scene coordinates, y up,
// [0,3]^3 for the training dam breaks); positions / velocities float3 (stride 12), density float (kg/m^3).
using System;
using UnityEngine;

public interface IGpuSimSource
{
    GraphicsBuffer PositionBuffer { get; }
    GraphicsBuffer VelocityBuffer { get; }
    GraphicsBuffer DensityBuffer { get; }
    int GpuCount { get; }
    /// <summary>Simulation time of the state the buffers hold (estimate for solvers whose dt lives on the GPU).</summary>
    float SolverTime { get; }
    /// <summary>"Solver frame" rate: OnSolverStepGpu fires once per 1 / SimHz of simulation time (the emitter's frame).</summary>
    float SimHz { get; }
    Vector3 Gravity { get; }
    Vector3 DomainMin { get; }
    Vector3 DomainMax { get; }
    Vector2 DomainCenterXZ { get; }
    LiveSphProvider.Obstacle[] SceneObstacles { get; }
    Transform transform { get; }
    float LastStepMs { get; }
    event Action<int, float> OnSolverStepGpu;
}

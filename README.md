# ElevatorHarness

[中文说明](README.zh-CN.md)

A robot- and agent-independent harness for visually guided panel interaction:
**observe global/local views → align in a plane → approach/press along its normal → inspect feedback → withdraw or retry**.
Tool configurations, including gripper opening and closing, change both contact geometry and visibility.

Elevator button pressing is the first task. Navigation, riding an elevator, high-frequency VLA control, and arbitrary manipulation are outside this release.

<img width="1196" height="699" alt="image" src="https://github.com/user-attachments/assets/ca22685d-0aeb-419d-a0de-271d8401b0d5" />


## Quick start

Python 3.10 or later. The core has no third-party runtime dependencies and imports no robot SDK.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
elevator-harness simulate --output /tmp/elevator-demo
elevator-harness simulate --rotated --output /tmp/elevator-demo-rotated
```

Each output directory must be new. These examples use a deterministic robot fixture and a scripted Agent: they test integration, **not physical performance**. The rotated example changes the operation axes, tool names, and camera mobility without changing the Harness.

## Design

```text
Task + capabilities + observation
                 ↓
              Agent
                 ↓ action proposal bound to observation ID
       ExecutionPolicy
                 ↓ allowed action, unchanged
          RobotAdapter
                 ↓ execution receipt + fresh observation
             Harness ──→ journal, images, verification
```

- `OperationFrame` defines orthogonal unit `u/v/n` axes in a named robot frame. Motion is `[du,dv,dn]` in metres; positive `n` approaches the panel. View angles are absolute radians. The S1 adapter converts these to SDK millimetres/degrees.
- `Capabilities` advertises cameras, movable views and tool configurations. Unsupported view control is absent from the model's action schema.
- `Observation` binds images, depth, measured state, tool configuration and calibration revision to one ID.
- `Agent` proposes actions; `ExecutionPolicy` accepts or rejects them without clipping. `EndToEndAgent` uses current images directly. `LocalizedAgent` uses dual-view coarse localization, local crop refinement, and closed-tool direct decisions with a current global zoom; repeated-motion reviews and press-attempt memory preserve feedback.
- `PressButtonTask` tracks reversible task stages. `CalibratedPressPolicy` provides optional overlap checks, an explicitly transferred open-tool range, staged approach, path review, withdrawal and attempt limits.
- `Verifier` distinguishes completion evidence from the Agent's completion claim. Visual verification remains a **model judgment**, never human confirmation.

## Use the Python API

```python
from elevator_harness import Harness, PressButtonTask, Recorder
from elevator_harness.simulation import SimulatedRobot, ScriptedAgent, SimulatedVerifier

run = Harness(
    robot=SimulatedRobot(),
    agent=ScriptedAgent(),
    task=PressButtonTask("activate the synthetic button and withdraw"),
    verifier=SimulatedVerifier(),
    recorder=Recorder("/tmp/my-new-run"),
)
result = run.run()
print(result.status, result.verification)
```

For coarse-to-fine localization, first install `pip install 'elevator-harness[vision]'`. See the [perception pipeline](docs/perception-pipeline.md) for crop geometry, stage switching, fallback and tuning.

For a visual Agent, inject `ChatCompletionsClient(base_url, model, key_env=...)` into `EndToEndAgent` or `LocalizedAgent` and `VisualVerifier`. The base includes your API prefix (for example, `/v1`); the client appends `/chat/completions`. Your endpoint must support image inputs and the supplied function schemas. There are no built-in model names or vendor capability guarantees. Credentials are read from the named environment variable and excluded from recordings; HTTP redirects are rejected.

`LocalizedAgent` defaults to 3× local crops enlarged to 768 pixels and 6× current global crops during calibrated pressing. Crop coordinates are mapped back before action planning or overlap checks; invalid head pose makes the global crop unavailable rather than reusing old pixels.

`LocalizedAgent` can be paired with `OverlapPolicy` or a calibrated `CalibratedPressPolicy`. An end-to-end Agent does not implicitly enable those visual gates. All modes still validate action structure, device capabilities and current-observation binding.

## Optional S1 example

The [S1 integration](examples/s1/README.md) includes a robot-local camera/SDK service and a client using this Harness. It is included in the source distribution, not the core wheel. Hardware dependencies are installed separately. A local calibration must be filled in before hardware acquisition; no measured deployment calibration or site photos are shipped.

## Results and limits

Runs retain image files and an append-only `events.jsonl` journal containing proposals, policy decisions, actual execution results, model requests/responses and verification. `completed` means the selected verifier confirmed its criterion; inspect its `source` before interpreting the result. `unverified` preserves an Agent completion claim with denied or insufficient evidence.

Defaults: 40 decision rounds, 120-second model/robot operation timeouts, at most 3 output corrections per round, and 8 history transitions supplied to the Agent. Every fresh run gets independent state. No restart automatically resumes or replays a command. Device adapters must implement bounded, cancellable operations; an arbitrary blocking third-party adapter cannot be forcibly made cancellable by the runtime.

Recorded image replay can compare perception and decisions. It cannot supply the world response to a newly proposed action and is not a closed-loop success benchmark. Two-dimensional image overlap is not proof of three-dimensional alignment or contact. A sent command and a settled measurement are distinct from verified target arrival.

## Development and packaging

```bash
python -m pip install -e '.[test,build]'
python -m pytest -q
python -m build
python -m twine check dist/*
```

Tests use synthetic cameras, a local fake model endpoint and fake SDKs; they never command a real robot. See [architecture and contracts](docs/architecture.md), [extension guide](docs/extending.md), and [release validation](docs/validation.md).

MIT license for project code. Astribot, ROS, camera drivers and other external dependencies retain their own licenses and are not bundled. This public repository starts from a clean ElevatorHarness snapshot. The earlier deployment repository and its history remain separate and are not included here.

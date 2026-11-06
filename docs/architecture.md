# Architecture and contracts

## Ownership

Each `Harness` is single-use and owns an episode's cancellation event, action ID set, observations and transition history. Agent caches, task state and press-policy state are instance-local. Construct new instances for each run. No module-global model selection or calibration exists.

The robot owns sensing and actuation. The runtime invokes a single operation at a time and checks cancellation after a model reply and before execution. An adapter must honor its timeout/cancellation argument, serialize physical writers and provide bounded `stop()` and `close()`. The supplied model transport returns promptly on cancellation and discards late network responses; it does not pretend to cancel computation already underway on a server.

## Data flow

1. Capture a fresh observation and verify its configuration/calibration against capabilities.
2. Build context with current observation, task progress, bounded actual-action history and policy state.
3. Ask the Agent for one proposal. Localized agents refine a local crop during alignment, then switch to direct visual assessment and a current global crop during calibrated pressing; the full record retains perception and independent progress reviews.
4. Validate action fields, capabilities, current observation and command identity. Apply the execution policy. Rejections are fed back for a bounded correction; commands are never silently clipped.
5. Journal admission before execution. Execute once; distinguish acceptance, sending, measured feedback and capture status. A post-motion capture failure terminates this run and never repeats the movement.
6. Record actual feedback, update task/policy state, and capture a new observation after every motion or configuration change.
7. On `done`, run the verifier separately. Stop and close the adapter on all terminal paths. Restarting requires a new run directory and initial observation.

## Coordinates and tools

`OperationFrame(reference,u,v,n)` contains orthogonal unit basis vectors in the robot frame. `[du,dv,dn]` may contain all three components. `+n` approaches the panel. Core distance units are metres; angles are radians. End-effector orientation is preserved in this release. Frame estimation and robot kinematics belong to the adapter/calibration workflow.

`ToolConfiguration` specifies a name, physical contact reference, calibration revision, optional normalized local contact pixel and visibility description. Camera pixels are normalized to `[0,1]` (x right, y down). They are not metric displacements. Configuration changes require new evidence. A policy can explicitly carry a frozen measurement across a named open-to-press transition, with its observation ID, revision and source retained.

`Observation.images` maps roles to image bytes/MIME types. Global and local are the built-in visual roles; adapters can expose other roles to custom Agents. Depth is optional and has explicit validity. A sensor adapter must make image/state freshness and measurement provenance meaningful; the core cannot infer sensor synchronization from an ID.

## Press policy

Open-tool closing requires a fully visible target covering the configured merged contact pixel and a valid in-range depth. The admitted observation's range is frozen only after the close command is sent. Forward allowance is bounded by the frozen distance minus the contact target, a maximum stroke, and actual normal travel when available. Current valid depth can further reduce, never enlarge, the allowance. Without live depth the frozen bound is used. Calibration changes invalidate the frozen measurement.

The first approach leaves a standoff. Further pressing requires a current global/local path assessment when configured. During direct pressing this assessment comes from the action response itself, without a separate localization call. The live allowance is supplied before the decision. A completed pressing step requires withdrawal before reopening; reopening discards the old frozen range. The attempt budget applies to new closed-tool attempts. Retreat, observation and reporting inability remain available. These checks are explicit rules, not proof of contact or a hardware safety controller.

## Completion and records

The button verifier requires distinct ordered observations: initial light off, later activation, and a subsequent image with the target still lit and the hand clear. Missing evidence returns `unknown`; a final unlit target returns `denied`. The `model_visual_judgment` source is distinct from `simulation_fixture` or a future human/sensor verifier.

Run events are JSON Lines. Observations save raw images under hashed observation IDs; request/response records preserve model evidence. Credentials and HTTP authorization headers are never intentionally logged; configured API key strings are redacted from model response/request records as well. Treat all recorded images and prompts as your data when distributing runs.

Output correction retries are bounded; HTTP model transport failures terminate instead of repeating model calls. The S1 HTTP transport may retry a request with the exact same command ID on ambiguous delivery, and its service deduplicates that ID before moving.

Detailed crop transforms, stage-dependent evidence, review predictions and attempt memory are described in [the perception pipeline](perception-pipeline.md).

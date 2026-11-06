# Adding a robot, Agent or task

## Robot adapter

Implement `RobotAdapter` from `elevator_harness.types`. Provide `capabilities`, `observe(cancel,timeout)`, `execute(action,cancel,timeout)`, `stop()` and `close()`.

1. Declare the manipulator, operation basis, camera roles, movable views, available tool configurations and depth support. Never advertise an absent head actuator.
2. Convert metre-based operation deltas with `capabilities.frame.to_robot(delta)`, then convert units at the SDK boundary. Camera calibration and local contact pixels must be measured for this mounting.
3. Return fresh uniquely identified observations containing synchronized measured feedback. Include `state.position_m` in the declared robot reference frame if a calibrated press policy should use measured normal travel.
4. Serialize physical writes. Bind actions to the observation actually used, deduplicate command IDs, and return a distinct execution receipt. Do not translate a motion/capture failure into success or replay a partially executed action.
5. Implement bounded cancellation and cleanup; keep hardware imports out of module initialization when possible. Start with synthetic contract tests and validate actual robot behavior separately.

An ALOHA adapter would supply its own kinematics, tool configurations and global/local camera mapping. This release does not include or claim an ALOHA integration. A second arm is not implicitly coordinated; one adapter controls one selected manipulator in v1.

## Agent

Implement `decide(context,cancel) -> ActionProposal`. Input contains the task goal, latest observation, capabilities, task state, bounded history, optional perception evidence and rejection feedback. Only actual executed transitions enter history. Bind the proposal to `context.observation.id` and use a new command ID.

Actions: `move(delta=[du,dv,dn])`, `configure_tool(configuration=name)`, `adjust_view(camera=role,angles=[yaw,pitch])`, `observe`, `done`, `unable`. A different robot may expose view joints with different semantics; document those in its adapter and use a custom Agent when two absolute angles are not appropriate. Arbitrary joint trajectories and asynchronous VLA chunks require a future execution contract.

The built-in visual Agents take a `ModelClient` with `complete(messages,tool,cancel) -> dict`: return decoded arguments for the one requested tool. The supplied Chat Completions transport is one implementation. Inject a client for another API instead of changing the Harness. Model IDs and transport dialects are deliberately not inferred from brand names.

## Policy and verification

Implement `check(action,context) -> Decision`, `record(action,before,result)` and `describe()`. `check` may prepare an admission-bound measurement, but execution effects belong in `record`; rejected actions must not consume travel or attempts. An optional `prepare(context)` hook updates current sensor-based allowances before the Agent is called. `CompositePolicy` delegates preparation and combines checks with AND semantics. Do not combine a closed-tool occlusion-aware press policy with an unconditional visible-tip rule unless that is intended.

For another panel task, supply a task and verifier appropriate to its actual feedback. A touchscreen may verify a UI transition instead of a light. A rotary knob requires additional action support. Human review can be implemented as a separate verifier; never relabel a model verdict as human verification.

## Recorded re-decision

`replay_decision(source, observation_id, fresh_agent, recorder)` reconstructs a saved observation and its actual prior transition history, then calls the Agent once without a robot. Results are explicitly tagged `replay_only`. It does not execute the policy or a new action, advance to subsequent original frames as if they followed the new proposal, or assign a success score. Use it to inspect perception and proposed actions, not to certify physical behavior.

# Coarse-to-fine localization and closed-tool pressing

The built-in `LocalizedAgent` now preserves the visual mechanisms from the S1 deployment as configurable stages. They are not specific to a model brand. The model endpoint still needs image inputs and the requested tool schemas.

Install image-transform support with `pip install 'elevator-harness[vision]'` (or `pip install -e '.[vision]'` from source). The core, scripted simulation and end-to-end Agent remain usable without Pillow. A crop-enabled LocalizedAgent checks the optional dependency at construction, before the S1 client connects to hardware.

## Alignment stage

1. One request independently localizes the target and physical contact reference in the full **global and local** images. It records visibility and declared coordinate units. Both `normalized_01` and `normalized_1000` are converted to canonical full-image `[0,1]` coordinates; pixels or mixed units are not assumed.
2. Around the coarse local target box, form a square crop with side **3 times the larger box dimension**, constrained to the source image. Resize its longest side to **768 pixels**.
3. Make a separate `locate_button_in_crop` request. Its returned box belongs to the **crop**, and is mapped through the original pixel crop window back to full-image coordinates. Resized display pixels are not used as robot coordinates.
4. Keep the coarse result and refinement lineage. The refined target box replaces the canonical box consumed by action context, overlap checks and closing/range checks; they do not use inconsistent rectangles.
5. A failed, malformed or unknown second pass preserves the coarse result and logs a fallback. Cancellation aborts the decision. A returned partial box remains partial and cannot pass a full-visibility gate. The physical contact reference retains its full-image identity; the refinement only replaces the target face box.

The global box is kept as a coarse estimate during alignment; it is not subjected to an extra refinement request. Geometry advice supplies raw offsets, optional mounting-specific u/v signs, and guidance to maintain a usable global view. Signs are not pixel-to-metre calibration or execution permission. Partial or uncertain evidence is not silently promoted to certainty.

## Closed-tool press stage

With a range frozen by `CalibratedPressPolicy` and the matching press tool active, the Agent stops separate localization and coordinate-derived direction hints. It uses both current full images directly to assess the closed contact end, button and indicator. For use without that policy, `PerceptionConfig.press_tool` can explicitly select the direct-press configuration.

The Agent may append a **6-times** crop of the **current global image** around the most recent pre-press global target box. The box is an ROI hint, not a current location measurement. Full images remain in the request. Each crop records separate current-image and historical-box observation IDs, source hash, image size, crop window and resized size.

For a movable global camera, measured view angles are required. A change greater than **1 degree** from the anchor invalidates the crop. Explicit view adjustment, calibration changes, optional camera-pose revision changes, changed image geometry, missing images or missing pose evidence also skip it, with a recorded reason. A fixed-camera adapter may omit joint angles, but must update its pose revision if its physical viewpoint moves. The Agent then uses the full current views; it does not reuse old crop pixels as current evidence.

The action response includes a current `press_assessment` describing both views, conflict and path clarity. The policy checks this assessment before final pressing. It does not add a separate localization call. Current measured travel/live-depth allowance is prepared before the decision. Final pressing is normal-only by default; in-plane corrections are separate actions. Reopening returns to coarse/fine localization and clears the old crop anchor before obtaining new geometry.

## Feedback and recovery

- Repeated executed u/v adjustments in the closed stage trigger an independent BEFORE/CURRENT review, including crops made from each respective image when valid. In the alignment stage, repeated forward movement or in-plane movement without consistent projection improvement triggers review.
- Three same-direction moves is the default threshold. Observation-only rounds do not increment or clear the streak. Tool/view changes end a comparable interval; the same command window is reviewed once. New executed motion can trigger a new window.
- Changes to box scale or contact/configuration identity make apparent image improvement uncertain. Review prompts explicitly consider parallax, object identity, passing the target and alternative explanations.
- After a review, the next action response must include a current hypothesis, expected visual change and a fallback if that prediction fails. These remain model claims; they do not generate robot motion automatically.
- The last three press attempts retain close observations, original localization, cumulative u/v adjustments, deepest executed-forward evidence, withdrawal/reopening and prior alignment claims. Relevant historical images are explicitly labelled historical.
- Each closed-tool in-plane adjustment supplies its actual before/after states, measured displacement when available and settling feedback to the next decision. Reopening or reaching a depth bound is not automatically labelled failed contact.

## Configuration

```python
from elevator_harness import LocalizedAgent, PerceptionConfig

agent = LocalizedAgent(model, perception=PerceptionConfig(
    refine_local=True,
    refinement_scale=3.0,
    crop_side_px=768,
    press_zoom=True,
    press_zoom_scale=6.0,
    direct_press=True,
    repeat_count=3,
    attempt_memory_limit=3,
    # Set only after validating this exact camera mounting and tool orientation:
    local_axis_signs=None,
))
```

The S1 example includes a `perception` section enabling these stages. Its local direction signs `[1,1]` mean target left → +u and target above → +v for the original mounting; verify or change this calibration for your setup. Generic defaults do not assume these signs. Cropping and direct-stage switching can be individually disabled.

## Migration evidence

| Deployment mechanism | Harness location / evidence |
|---|---|
| Dual-view coarse locations and coordinate normalization | `agents.inspect(..., dual=True)` |
| Local 3× crop, 768-pixel enlargement, second-pass localization | `PerceptionPipeline.localize` |
| Crop-to-full mapping, fallback, visibility propagation | `vision.restore_box`, refinement records |
| One canonical refined box for planning and checks | `Context.evidence` |
| Closed-tool direct visual decisions | `LocalizedAgent`, current `press_assessment` |
| Current global 6× crop and pose-based invalidation | `PerceptionPipeline.global_zoom` |
| Executed-motion review and required reassessment | `feedback.motion_review_signal`, `LocalizedAgent` |
| Attempt memory and immediate correction feedback | `PressMemory` |
| Mount-specific direction and view advice | `geometry_advice`, `PerceptionConfig` |
| Inspectable image and decision lineage | `derived_images/` plus journal events |

These are software mechanisms tested with synthetic images, model responses and robot feedback. Physical task performance and a specific provider's model behavior still require separate validation. The migration does not claim identical outputs from the old model prompts or revive old dashboards, deployment paths or provider-specific global settings.

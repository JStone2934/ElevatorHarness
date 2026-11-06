# S1 example

This source-distribution example connects the generic Harness to an Astribot S1 right arm, head and gripper. It reuses the existing serialized SDK executor, measured state sampling and camera synchronization, including the task-polling lock and capture-skew retry fixes. It is not installed by the core wheel.

Run the module commands from the extracted source/repository root, after installing the core package normally. No `sys.path` modification is needed for the core. The optional SDK must be installed and licensed separately.

## Environment and calibration

On the robot, activate your Python environment with ROS-compatible numpy/OpenCV and the Astribot SDK available:

```bash
python -m pip install -e '.[s1,vision]'
export S1_SDK_ROOT=/path/to/your/astribot_sdk
source examples/s1/environment.sh
cp examples/s1/config.template.json /tmp/s1-local.json
```

The environment script sources the installed robot ROS setup rather than selecting an offline domain. `S1_ROS_ROOT`, `S1_ROS_SETUP` and `ASTRIBOT_CAMERA_INFO_PATH` can override installation defaults. OpenCV and ROS packages should come from the robot-compatible installation; they are not bundled in the core package.

Edit `/tmp/s1-local.json` before starting:

- Set a real calibration revision shared by both tools.
- Measure the **closed physical contact end** in the local image and fill `tools[1].local_contact_uv` and `depth_point` with the same `[x/(width-1), y/(height-1)]` value. These are not the sticker centre or a guessed panel centre.
- Measure `press.contact_depth_m` for this installation. Configure valid range bounds, maximum stroke, standoff, overpress and withdrawal for the actual setup. Shipped numerical limits are examples, not universal hardware guarantees.
- Confirm that the operation axes are expressed in `chassis`. The supplied mapping is `u=+Y`, `v=+Z`, `n=+X`. Change it if the panel normal differs.
- The `perception` section enables local coarse/fine refinement and closed-tool direct decisions with global zoom. Verify `local_axis_signs` for this mounting; the default `[1,1]` means target image-left → +u and image-above → +v. Head angles are exported as radians for crop-anchor validity; the 1-degree threshold is configurable.
- Configure camera topics and camera SDK header location as needed. The sample roles map head RGB to global and right-wrist RGB/depth to local.

The template deliberately contains null calibration fields and cannot start hardware. `localized` also requires matching depth/contact pixels and a calibrated contact distance. `end_to_end` uses the same calibrated observation stream without the localized execution rules. No private calibration or photos ship with the project.

## Start

Robot terminal:

```bash
python -m examples.s1.serve --config /tmp/s1-local.json --output /tmp/s1-service
```

The SDK's normal control-rights prompt remains interactive; `--auto-confirm` explicitly accepts it. The service validates the config first, starts the camera process, obtains control, and exposes loopback port 6210. Camera port 8765 exposes snapshot/capture APIs; the old interactive dashboard is not retained. Exit with Ctrl-C or `q` followed by Enter.

Client terminal on the robot, or on a workstation with the same source distribution and identical config:

```bash
# Workstation only, in a separate terminal:
ssh -N -L 6210:127.0.0.1:6210 S1B

# Use the endpoint/model/key variable appropriate to your deployment:
python -m examples.s1.run \
  --config /tmp/s1-local.json \
  --robot-base http://127.0.0.1:6210 \
  --model-base http://localhost:8000/v1 --model your-vision-model \
  --goal 'Press the elevator down-call button and withdraw' \
  --output /tmp/s1-harness-run
```

For authenticated APIs, export the key separately and pass `--key-env YOUR_KEY_VARIABLE`. No model or provider is hard-coded. The model sees global/local images, measured state and the chosen tool descriptions. Camera roles do not imply a fixed image-to-motion direction table.

Service and client compare a configuration hash. Measured gripper positions must identify open or closed; an intermediate position produces an observation error rather than borrowing an old configuration. Settling is best-effort and does not prove target arrival. The legacy service continues to distinguish motion completion from capture/storage failure; the Harness records the receipt and does not repeat a sent action when capture fails.

## Hardware acceptance (manual, not performed by automated tests)

1. Confirm camera roles, fresh timestamps, frame orientation, measurement point, calibration and current gripper state.
2. Validate small positive/negative `u/v/n` movements and measured displacement with the installed SDK.
3. Check that closing/reopening changes observations and that frozen measurements are only carried across the intended transition.
4. Exercise localized standoff, pre-contact review, final press, withdrawal and a failed-press retry. Inspect both requested and measured motion.
5. Test terminal cancellation while waiting for the model and while moving. Confirm SDK cleanup and no late action.
6. Review initial lamp, activation and withdrawal evidence separately from model conclusions. Record human verification outside the model verdict.

See [perception migration details](../../docs/perception-pipeline.md) for the complete stage sequence, image lineage, retry memory and review triggers.

The offline tests validate software integration with fake cameras and SDKs; they do not establish successful physical button operation.

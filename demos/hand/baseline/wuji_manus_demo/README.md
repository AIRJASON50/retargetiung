# Wuji Manus Retargeting Demo

Minimal standalone demo: replay Manus glove data with live IK retargeting in MuJoCo.

## Pipeline

```
manus1.pkl (MediaPipe landmarks, 21x3 per frame)
  -> InputDataReplay (playback by timestamp)
  -> Retargeter.retarget() (IK solve: landmarks -> 20-dim joint angles)
  -> MuJoCo viewer (visualize)
```

This is **live retargeting**, not pre-computed joint angle playback.
Each frame's MediaPipe landmarks are fed through the IK solver in real time.

## Files

```
wuji_manus_demo/
├── play_manus.py                      # Main script
├── config/retarget_manus_left.yaml    # Manus left hand IK config
├── data/manus1.pkl                    # 13341 frames, left hand only
└── input_devices/                     # Minimal replay module
    ├── __init__.py
    ├── base.py                        # InputDeviceBase ABC
    └── input_data_replay.py           # PKL reader with timestamp-based playback
```

## Usage

```bash
cd /home/l/ws/RL/retargeting/wuji_manus_demo
python play_manus.py
python play_manus.py --speed 2.0              # 2x playback
python play_manus.py --mjcf /path/to/left.xml # custom MJCF
```

## Dependencies

- `wuji-retargeting` (installed via `pip install -e .../wuji_retargeting_private/public/`)
- `mujoco`
- `numpy`

## Notes

- `manus1.pkl` contains left hand data only (`right_fingers` is all zeros)
- Default MJCF: `/home/l/ws/doc/WujiRepo/urdf_cali/reference/result/xml/left.xml`
- Source: extracted from `wuji_retargeting_private/test/example_test/`

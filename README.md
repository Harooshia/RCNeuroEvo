# RCNeuroEvo

Code Bullet–style neuroevolution racing simulation in Python using **TensorFlow/Keras** and **Tkinter** rendering.

## Features

- 2D top-down **closed-loop race track** with inner/outer boundaries, varied lane width, centerline, and curbs.
- Rectangle car physics:
  - acceleration
  - friction
  - steering and rotation coupled to speed
- Collision detection against track boundaries.
- 7-ray sensor semicircle in front of each car.
- TensorFlow feed-forward policy network:
  - inputs = ray distances + normalized speed
  - outputs = steering (`tanh`, -1..1) and throttle (`sigmoid`, 0..1)
- Genetic algorithm evolution (no backprop):
  - elite selection (top 20%)
  - uniform crossover for model weights
  - Gaussian mutation noise
- Enhanced Tkinter GUI with improved HUD panel, speed bar, centerline markers, and curb styling.
- Generation-by-generation visualization with best car highlighting.
- Fast training mode with rendering disabled.
- Save/load best weights for watch mode.

## Install

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install this project (editable)

```bash
pip install -e .
```

If you prefer direct dependencies:

```bash
pip install numpy tensorflow
```

## Quick test (sanity check)

Run from the **repository root** (the folder that contains `neuroevo/`):

```bash
python -c "import neuroevo; print('neuroevo import ok')"
```

Then run a tiny no-render training smoke test:

```bash
python -m neuroevo.main --no-render --population 10 --generations 2 --max-steps 100 --speed 5
```

## Train normally (Tkinter rendering)

```bash
python -m neuroevo.main --population 60 --generations 200 --max-steps 1600 --speed 5
```

### Useful flags

- `--no-render`: disables visualization for fast training
- `--show-sensors`: draws sensor rays for best car
- `--save-path models/best.weights.h5`: output weights path

## Watch best model

```bash
python -m neuroevo.main --load-path models/best.weights.h5 --show-sensors
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'neuroevo'`

This means Python cannot find the package in your environment.

Use one of these fixes:

1. Run commands from repo root and install editable package:

```bash
cd /path/to/RCNeuroEvo
pip install -e .
python -m neuroevo.main --no-render --population 10 --generations 1 --max-steps 50
```

2. Or set `PYTHONPATH` explicitly:

```bash
PYTHONPATH=. python -m neuroevo.main --no-render
```

3. Verify interpreter and pip point to the same environment:

```bash
which python
python -m pip --version
python -c "import sys; print(sys.executable)"
```


### Still seeing `import pygame` in traceback?

You are likely running an older installed copy of the package. Confirm the module path:

```bash
python -c "import neuroevo.main as m; print(m.__file__)"
```

If it does **not** point to your current repo checkout, reinstall and retry:

```bash
pip uninstall -y rc-neuroevo neuroevo || true
pip install -e .
python -m neuroevo.main --no-render --population 10 --generations 1 --max-steps 50
```

## Notes

- The system intentionally avoids gradient-based training. All learning comes from selection + crossover + mutation of TensorFlow model weights.
- Default input size is `8` (7 sensors + speed).

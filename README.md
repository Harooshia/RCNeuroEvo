# RCNeuroEvo

Code Bullet–style neuroevolution racing simulation in Python using **Pygame** and **TensorFlow/Keras**.

## Features

- 2D top-down **closed-loop race track** with inner/outer boundaries.
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
- Generation-by-generation visualization with best car highlighting.
- Fast training mode with rendering disabled.
- Save/load best weights for watch mode.

## Install

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install numpy pygame tensorflow
```

## Train

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

## Notes

- The system intentionally avoids gradient-based training. All learning comes from selection + crossover + mutation of TensorFlow model weights.
- Default input size is `8` (7 sensors + speed).

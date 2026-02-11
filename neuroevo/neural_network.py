"""TensorFlow model wrapper with helper methods for neuroevolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


@dataclass
class NeuralNetwork:
    """Two-head policy network yielding steering and throttle."""

    input_size: int
    hidden_sizes: Tuple[int, ...] = (16, 16)

    def __post_init__(self) -> None:
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        inp = keras.Input(shape=(self.input_size,), dtype=tf.float32)
        x = inp
        for h in self.hidden_sizes:
            x = keras.layers.Dense(h, activation="relu")(x)
        steering = keras.layers.Dense(1, activation="tanh", name="steering")(x)
        throttle = keras.layers.Dense(1, activation="sigmoid", name="throttle")(x)
        out = keras.layers.Concatenate(name="action")([steering, throttle])
        model = keras.Model(inputs=inp, outputs=out)
        # No training via gradient descent, but compile keeps API ergonomic.
        model.compile(optimizer="adam", loss="mse")
        return model

    def clone(self) -> "NeuralNetwork":
        clone = NeuralNetwork(self.input_size, self.hidden_sizes)
        clone.set_weights(self.get_weights())
        return clone

    def predict_action(self, state: np.ndarray) -> Tuple[float, float]:
        x = state.reshape(1, -1).astype(np.float32)
        pred = self.model(x, training=False).numpy()[0]
        steering = float(np.clip(pred[0], -1.0, 1.0))
        throttle = float(np.clip(pred[1], 0.0, 1.0))
        return steering, throttle

    def get_weights(self) -> List[np.ndarray]:
        return [np.array(w, copy=True) for w in self.model.get_weights()]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        self.model.set_weights(weights)

    def save_weights(self, path: str) -> None:
        self.model.save_weights(path)

    def load_weights(self, path: str) -> None:
        self.model.load_weights(path)

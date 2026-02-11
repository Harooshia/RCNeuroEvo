"""Car agent physics, sensors, and fitness logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pygame

from .track import Track


@dataclass
class Car:
    """Physics state and behavior for a single evolving car."""

    x: float
    y: float
    angle: float
    width: float = 12
    length: float = 22
    max_speed: float = 6.0
    acceleration_rate: float = 0.2
    friction: float = 0.03
    steer_strength: float = 0.06
    sensor_range: float = 140.0
    sensor_angles: Tuple[float, ...] = (-1.2, -0.6, -0.2, 0.0, 0.2, 0.6, 1.2)
    speed: float = 0.0
    alive: bool = True
    total_distance: float = 0.0
    time_alive: int = 0
    checkpoints_passed: int = 0
    next_checkpoint: int = 0
    fitness: float = 0.0
    last_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    def __post_init__(self) -> None:
        self.last_position = np.array([self.x, self.y], dtype=np.float32)

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def reset(self, x: float, y: float, angle: float) -> None:
        """Reset state for next generation."""
        self.x, self.y, self.angle = x, y, angle
        self.speed = 0.0
        self.alive = True
        self.total_distance = 0.0
        self.time_alive = 0
        self.checkpoints_passed = 0
        self.next_checkpoint = 0
        self.fitness = 0.0
        self.last_position = np.array([x, y], dtype=np.float32)

    def step(self, steering: float, throttle: float, track: Track) -> None:
        """Advance car simulation using control outputs."""
        if not self.alive:
            return

        steering = float(np.clip(steering, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))

        self.speed += throttle * self.acceleration_rate
        self.speed -= self.friction
        self.speed = float(np.clip(self.speed, 0.0, self.max_speed))

        # Steering causes rotation proportionally to current speed.
        self.angle += steering * self.steer_strength * (0.2 + self.speed / self.max_speed)

        direction = np.array([np.cos(self.angle), np.sin(self.angle)], dtype=np.float32)
        prev_pos = self.pos
        new_pos = prev_pos + direction * self.speed
        self.x, self.y = float(new_pos[0]), float(new_pos[1])

        self.time_alive += 1
        segment_distance = float(np.linalg.norm(new_pos - prev_pos))
        self.total_distance += segment_distance

        if not self._corners_on_track(track):
            self.alive = False
            self.fitness = self.compute_fitness(crashed=True)
            return

        self._update_checkpoint_progress(prev_pos, new_pos, track)
        self.fitness = self.compute_fitness(crashed=False)
        self.last_position = new_pos

    def _car_corners(self) -> List[Tuple[float, float]]:
        """Return world-space coordinates of the 4 rectangle corners."""
        half_l = self.length / 2
        half_w = self.width / 2
        local = np.array(
            [
                [half_l, -half_w],
                [half_l, half_w],
                [-half_l, half_w],
                [-half_l, -half_w],
            ],
            dtype=np.float32,
        )
        rot = np.array(
            [[np.cos(self.angle), -np.sin(self.angle)], [np.sin(self.angle), np.cos(self.angle)]], dtype=np.float32
        )
        corners = local @ rot.T + self.pos
        return [(float(c[0]), float(c[1])) for c in corners]

    def _corners_on_track(self, track: Track) -> bool:
        for x, y in self._car_corners():
            if not track.is_on_track(x, y):
                return False
        return True

    def _update_checkpoint_progress(self, prev_pos: np.ndarray, new_pos: np.ndarray, track: Track) -> None:
        checkpoint = track.checkpoints[self.next_checkpoint]
        if track.segment_intersection(
            (float(prev_pos[0]), float(prev_pos[1])),
            (float(new_pos[0]), float(new_pos[1])),
            checkpoint.start,
            checkpoint.end,
        ):
            self.checkpoints_passed += 1
            self.next_checkpoint = (self.next_checkpoint + 1) % len(track.checkpoints)

    def compute_fitness(self, crashed: bool) -> float:
        """Fitness combines distance, survival, checkpoint reward, and crash penalty."""
        fitness = self.total_distance + 0.25 * self.time_alive + 200.0 * self.checkpoints_passed
        if crashed:
            fitness -= 100.0
        return float(max(fitness, 0.0))

    def get_sensor_readings(self, track: Track) -> np.ndarray:
        """Raycast sensor distances in [0,1], where 1 means max range clear."""
        readings = []
        origin = self.pos
        for offset in self.sensor_angles:
            ray_angle = self.angle + offset
            direction = np.array([np.cos(ray_angle), np.sin(ray_angle)], dtype=np.float32)
            dist = self.sensor_range
            for d in np.linspace(0, self.sensor_range, 45):
                point = origin + direction * d
                if not track.is_on_track(float(point[0]), float(point[1])):
                    dist = float(d)
                    break
            readings.append(dist / self.sensor_range)
        return np.array(readings, dtype=np.float32)

    def get_network_input(self, track: Track) -> np.ndarray:
        sensors = self.get_sensor_readings(track)
        speed_norm = np.array([self.speed / self.max_speed], dtype=np.float32)
        return np.concatenate([sensors, speed_norm], axis=0)

    def draw(self, screen: pygame.Surface, track: Track, best: bool = False, draw_sensors: bool = False) -> None:
        if not self.alive:
            return
        color = (255, 220, 50) if best else (50, 180, 250)
        corners = self._car_corners()
        pygame.draw.polygon(screen, color, corners)

        if draw_sensors:
            origin = (self.x, self.y)
            readings = self.get_sensor_readings(track)
            for ratio, offset in zip(readings, self.sensor_angles):
                angle = self.angle + offset
                dist = float(ratio * self.sensor_range)
                end = (self.x + np.cos(angle) * dist, self.y + np.sin(angle) * dist)
                pygame.draw.line(screen, (255, 100, 100), origin, end, 1)

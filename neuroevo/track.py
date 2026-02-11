"""Track geometry and collision/checkpoint helpers for the racing simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pygame


Point = Tuple[float, float]


@dataclass
class Checkpoint:
    """Represents a checkpoint line segment between outer and inner boundaries."""

    start: Point
    end: Point


class Track:
    """Closed-loop track represented by a drivable mask between two polygons."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.center = np.array([width / 2, height / 2], dtype=np.float32)
        self.outer_points, self.inner_points = self._generate_loop_points()
        self.mask = self._build_drivable_mask()
        self.checkpoints = self._build_checkpoints()
        self.start_position = np.array(self.checkpoints[0].start, dtype=np.float32)
        first_cp = self.checkpoints[0]
        direction = np.array(first_cp.end, dtype=np.float32) - np.array(first_cp.start, dtype=np.float32)
        # Tangent direction roughly points along track progression.
        self.start_angle = float(np.arctan2(direction[1], direction[0]) + np.pi / 2)

    def _generate_loop_points(self, n_points: int = 36) -> Tuple[List[Point], List[Point]]:
        """Create a smooth noisy ring as outer/inner polygons."""
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        # Wavy radius profile to make the track non-trivial.
        base_radius = min(self.width, self.height) * 0.38
        wave = 1 + 0.15 * np.sin(3 * angles + 0.3) + 0.1 * np.sin(5 * angles - 0.7)
        outer_radius = base_radius * wave
        lane_width = min(self.width, self.height) * 0.12
        inner_radius = outer_radius - lane_width

        outer = []
        inner = []
        for angle, r_out, r_in in zip(angles, outer_radius, inner_radius):
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            outer.append((float(self.center[0] + cos_a * r_out), float(self.center[1] + sin_a * r_out)))
            inner.append((float(self.center[0] + cos_a * r_in), float(self.center[1] + sin_a * r_in)))
        return outer, inner

    def _build_drivable_mask(self) -> pygame.Mask:
        """Create a mask where drivable pixels are 1."""
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.polygon(surface, (255, 255, 255, 255), self.outer_points)
        pygame.draw.polygon(surface, (0, 0, 0, 255), self.inner_points)
        return pygame.mask.from_threshold(surface, (255, 255, 255, 255), (1, 1, 1, 255))

    def _build_checkpoints(self) -> List[Checkpoint]:
        """Checkpoint lines connecting corresponding points of outer and inner boundary."""
        checkpoints: List[Checkpoint] = []
        for outer, inner in zip(self.outer_points, self.inner_points):
            checkpoints.append(Checkpoint(start=outer, end=inner))
        return checkpoints

    def is_on_track(self, x: float, y: float) -> bool:
        """Check if a world coordinate lies on drivable area."""
        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
            return False
        return bool(self.mask.get_at((ix, iy)))

    def segment_intersection(self, p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
        """Return whether segments p1-p2 and q1-q2 intersect."""

        def orientation(a: Point, b: Point, c: Point) -> float:
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)
        return (o1 * o2 < 0) and (o3 * o4 < 0)

    def draw(self, screen: pygame.Surface) -> None:
        """Render track and checkpoints."""
        screen.fill((15, 15, 20))
        pygame.draw.polygon(screen, (120, 120, 120), self.outer_points)
        pygame.draw.polygon(screen, (15, 15, 20), self.inner_points)
        for cp in self.checkpoints:
            pygame.draw.line(screen, (55, 55, 80), cp.start, cp.end, 1)

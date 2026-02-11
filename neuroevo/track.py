"""Track geometry and collision/checkpoint helpers for the racing simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass
class Checkpoint:
    """Represents a checkpoint line segment between outer and inner boundaries."""

    start: Point
    end: Point


class Track:
    """Closed-loop track represented by outer and inner polygons."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.center = np.array([width / 2, height / 2], dtype=np.float32)

        self.outer_points, self.inner_points = self._generate_loop_points(n_points=72)
        self.center_points = self._build_centerline()
        self.checkpoints = self._build_checkpoints()

        self.start_position = np.array(self.center_points[0], dtype=np.float32)
        tangent = np.array(self.center_points[1], dtype=np.float32) - np.array(self.center_points[0], dtype=np.float32)
        self.start_angle = float(np.arctan2(tangent[1], tangent[0]))
        self.start_line = self.checkpoints[0]

    def _generate_loop_points(self, n_points: int) -> Tuple[List[Point], List[Point]]:
        """Create a smoother, more technical loop as outer/inner polygons."""
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        base_radius = min(self.width, self.height) * 0.34

        # Compose multiple harmonics for a more interesting course shape.
        wave = (
            1.0
            + 0.20 * np.sin(2 * angles + 0.35)
            + 0.13 * np.sin(3 * angles - 0.8)
            + 0.08 * np.sin(5 * angles + 1.7)
        )
        outer_radius = base_radius * wave

        # Dynamically vary lane width to introduce narrow and wide sections.
        lane_width = (min(self.width, self.height) * 0.11) * (0.85 + 0.25 * np.sin(4 * angles - 0.1))
        inner_radius = outer_radius - lane_width

        outer, inner = [], []
        for angle, r_out, r_in in zip(angles, outer_radius, inner_radius):
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            outer.append((float(self.center[0] + cos_a * r_out), float(self.center[1] + sin_a * r_out)))
            inner.append((float(self.center[0] + cos_a * r_in), float(self.center[1] + sin_a * r_in)))
        return outer, inner

    def _build_centerline(self) -> List[Point]:
        """Centerline is midpoint between outer and inner borders."""
        centerline: List[Point] = []
        for outer, inner in zip(self.outer_points, self.inner_points):
            centerline.append(((outer[0] + inner[0]) / 2.0, (outer[1] + inner[1]) / 2.0))
        return centerline

    def _build_checkpoints(self) -> List[Checkpoint]:
        checkpoints: List[Checkpoint] = []
        for outer, inner in zip(self.outer_points, self.inner_points):
            checkpoints.append(Checkpoint(start=outer, end=inner))
        return checkpoints

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: List[Point]) -> bool:
        """Ray-casting point-in-polygon test."""
        inside = False
        n = len(polygon)
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = (yi > y) != (yj > y)
            if intersects:
                x_at_y = (xj - xi) * (y - yi) / ((yj - yi) + 1e-9) + xi
                if x < x_at_y:
                    inside = not inside
            j = i
        return inside

    def is_on_track(self, x: float, y: float) -> bool:
        """Check if a world coordinate lies in drivable area."""
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        in_outer = self._point_in_polygon(x, y, self.outer_points)
        in_inner = self._point_in_polygon(x, y, self.inner_points)
        return in_outer and not in_inner

    @staticmethod
    def segment_intersection(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
        """Return whether segments p1-p2 and q1-q2 intersect."""

        def orientation(a: Point, b: Point, c: Point) -> float:
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)
        return (o1 * o2 < 0) and (o3 * o4 < 0)

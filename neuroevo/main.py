"""Main entrypoint for neuroevolution racing simulation."""

from __future__ import annotations

import argparse
import os
import time
import tkinter as tk

import numpy as np
import tensorflow as tf

from .car import Car
from .genetic_algorithm import GeneticAlgorithm
from .neural_network import NeuralNetwork
from .track import Track


class TkRenderer:
    """Simple Tkinter-based 2D renderer for track and cars."""

    def __init__(self, width: int, height: int, title: str = "Neuroevolution Racing") -> None:
        self.root = tk.Tk()
        self.root.title(title)
        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="#0f0f14", highlightthickness=0)
        self.canvas.pack()

    def _close(self) -> None:
        self.running = False
        self.root.destroy()

    @staticmethod
    def _flatten_points(points: list[tuple[float, float]]) -> list[float]:
        flat = []
        for x, y in points:
            flat.extend([x, y])
        return flat

    def draw_scene(
        self,
        track: Track,
        cars: list[Car],
        generation: int,
        step: int,
        alive: int,
        pop_size: int,
        best_fitness: float,
        global_best: float,
        speed: int,
        show_sensors: bool,
        best_car: Car | None,
    ) -> None:
        self.canvas.delete("all")

        # Track surfaces.
        self.canvas.create_polygon(
            *self._flatten_points(track.outer_points), fill="#787878", outline="#787878", width=2
        )
        self.canvas.create_polygon(
            *self._flatten_points(track.inner_points), fill="#0f0f14", outline="#0f0f14", width=2
        )

        # Checkpoints.
        for cp in track.checkpoints:
            self.canvas.create_line(cp.start[0], cp.start[1], cp.end[0], cp.end[1], fill="#3a3a55", width=1)

        # Cars.
        for car in cars:
            if not car.alive:
                continue
            fill = "#ffdc32" if (best_car is car) else "#32b4fa"
            self.canvas.create_polygon(*self._flatten_points(car.polygon_points()), fill=fill, outline=fill)

        if show_sensors and best_car is not None and best_car.alive:
            for start, end in best_car.sensor_lines(track):
                self.canvas.create_line(start[0], start[1], end[0], end[1], fill="#ff6464", width=1)

        overlay = [
            f"Gen: {generation}",
            f"Step: {step}",
            f"Alive: {alive}/{pop_size}",
            f"Best fitness: {best_fitness:.1f}",
            f"Global best: {global_best:.1f}" if global_best > -np.inf else "Global best: N/A",
            f"Speed x{speed}",
        ]
        for idx, line in enumerate(overlay):
            self.canvas.create_text(16, 16 + idx * 22, text=line, fill="#f5f5f5", anchor="nw", font=("Consolas", 13))

        self.root.update_idletasks()
        self.root.update()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Code Bullet-style neuroevolution racing sim")
    parser.add_argument("--population", type=int, default=60, help="Population size")
    parser.add_argument("--max-steps", type=int, default=1600, help="Max steps per generation")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations to train")
    parser.add_argument("--speed", type=int, default=1, help="Simulation speed multiplier (steps/frame)")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering for fast training")
    parser.add_argument("--show-sensors", action="store_true", help="Render sensor rays")
    parser.add_argument("--save-path", type=str, default="models/best.weights.h5", help="Where best weights are saved")
    parser.add_argument("--load-path", type=str, default="", help="Load weights and watch single car")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_watch_mode(track: Track, weight_path: str, show_sensors: bool) -> None:
    renderer = TkRenderer(track.width, track.height, title="Neuroevolution Racing (Watch)")

    input_size = 8
    net = NeuralNetwork(input_size=input_size)
    net.load_weights(weight_path)

    car = Car(x=float(track.start_position[0]), y=float(track.start_position[1]), angle=track.start_angle)

    step = 0
    while renderer.running:
        if car.alive:
            state = car.get_network_input(track)
            steer, throttle = net.predict_action(state)
            car.step(steer, throttle, track)

        renderer.draw_scene(
            track=track,
            cars=[car],
            generation=0,
            step=step,
            alive=1 if car.alive else 0,
            pop_size=1,
            best_fitness=car.fitness,
            global_best=car.fitness,
            speed=1,
            show_sensors=show_sensors,
            best_car=car,
        )
        step += 1
        time.sleep(1 / 60)


def run_training(args: argparse.Namespace) -> None:
    renderer = None if args.no_render else TkRenderer(1200, 800)

    track = Track(width=1200, height=800)
    ga = GeneticAlgorithm(
        population_size=args.population,
        input_size=8,
        start_x=float(track.start_position[0]),
        start_y=float(track.start_position[1]),
        start_angle=track.start_angle,
        elite_fraction=0.2,
        mutation_rate=0.1,
        mutation_scale=0.08,
        hidden_sizes=(24, 16),
    )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    global_best = -np.inf

    for gen in range(1, args.generations + 1):
        ga.reset_cars(float(track.start_position[0]), float(track.start_position[1]), track.start_angle)
        generation_start = time.time()

        for step in range(args.max_steps):
            if renderer is not None and not renderer.running:
                return

            for _ in range(args.speed):
                ga.step_population(track)
                if ga.living_count() == 0:
                    break

            if renderer is not None:
                best_genome = ga.best_genome()
                renderer.draw_scene(
                    track=track,
                    cars=[g.car for g in ga.genomes],
                    generation=gen,
                    step=step,
                    alive=ga.living_count(),
                    pop_size=args.population,
                    best_fitness=best_genome.fitness,
                    global_best=global_best,
                    speed=args.speed,
                    show_sensors=args.show_sensors,
                    best_car=best_genome.car,
                )
                time.sleep(1 / 60)

            if ga.living_count() == 0:
                break

        stats = ga.evolve()
        global_best = max(global_best, stats["best_fitness"])
        ga.save_best(args.save_path)

        elapsed = time.time() - generation_start
        print(
            f"Generation {gen:03d} | Best: {stats['best_fitness']:.2f} | "
            f"Elite Avg: {stats['avg_fitness']:.2f} | Time: {elapsed:.2f}s"
        )


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    track = Track(width=1200, height=800)

    if args.load_path:
        run_watch_mode(track, args.load_path, args.show_sensors)
    else:
        run_training(args)


if __name__ == "__main__":
    main()

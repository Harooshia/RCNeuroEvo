"""Main entrypoint for neuroevolution racing simulation."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pygame
import tensorflow as tf

from .genetic_algorithm import GeneticAlgorithm
from .neural_network import NeuralNetwork
from .track import Track


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
    """Load one model and run it alone for demonstration."""
    pygame.init()
    screen = pygame.display.set_mode((track.width, track.height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    input_size = 8  # 7 sensors + speed
    net = NeuralNetwork(input_size=input_size)
    net.load_weights(weight_path)

    from .car import Car

    car = Car(x=float(track.start_position[0]), y=float(track.start_position[1]), angle=track.start_angle)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if car.alive:
            state = car.get_network_input(track)
            steer, throttle = net.predict_action(state)
            car.step(steer, throttle, track)

        track.draw(screen)
        car.draw(screen, track, best=True, draw_sensors=show_sensors)
        status = "ALIVE" if car.alive else "CRASHED"
        text = font.render(f"WATCH MODE | Fitness: {car.fitness:.1f} | {status}", True, (240, 240, 240))
        screen.blit(text, (15, 12))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def run_training(args: argparse.Namespace) -> None:
    if not args.no_render:
        pygame.init()
        screen = pygame.display.set_mode((1200, 800))
        pygame.display.set_caption("Neuroevolution Racing")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 20)
    else:
        screen = None
        clock = None
        font = None

    track = Track(width=1200, height=800)
    input_size = 8

    ga = GeneticAlgorithm(
        population_size=args.population,
        input_size=input_size,
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
            if not args.no_render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            for _ in range(args.speed):
                ga.step_population(track)
                if ga.living_count() == 0:
                    break

            if not args.no_render and screen is not None and font is not None:
                track.draw(screen)
                best = ga.best_genome()
                for genome in ga.genomes:
                    genome.car.draw(
                        screen,
                        track,
                        best=(genome is best),
                        draw_sensors=(args.show_sensors and genome is best),
                    )

                overlay = [
                    f"Gen: {gen}",
                    f"Step: {step}",
                    f"Alive: {ga.living_count()}/{args.population}",
                    f"Best fitness: {best.fitness:.1f}",
                    f"Global best: {global_best:.1f}" if global_best > -np.inf else "Global best: N/A",
                    f"Speed x{args.speed}",
                ]
                for i, text in enumerate(overlay):
                    surf = font.render(text, True, (245, 245, 245))
                    screen.blit(surf, (16, 12 + i * 24))

                pygame.display.flip()
                clock.tick(60)

            if ga.living_count() == 0:
                break

        stats = ga.evolve()
        global_best = max(global_best, stats["best_fitness"])
        if ga.save_best(args.save_path):
            pass

        elapsed = time.time() - generation_start
        print(
            f"Generation {gen:03d} | Best: {stats['best_fitness']:.2f} | "
            f"Elite Avg: {stats['avg_fitness']:.2f} | "
            f"Alive end: {ga.living_count()} | Time: {elapsed:.2f}s"
        )

    if not args.no_render:
        pygame.quit()


if __name__ == "__main__":
    cli_args = parse_args()
    setup_seed(cli_args.seed)
    track_obj = Track(width=1200, height=800)

    if cli_args.load_path:
        run_watch_mode(track_obj, cli_args.load_path, cli_args.show_sensors)
    else:
        run_training(cli_args)

"""Genetic algorithm for evolving TensorFlow model weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .car import Car
from .neural_network import NeuralNetwork
from .track import Track


@dataclass
class Genome:
    """Encapsulates one policy network and its associated car."""

    network: NeuralNetwork
    car: Car
    fitness: float = 0.0


class GeneticAlgorithm:
    """Population management, selection, crossover, and mutation."""

    def __init__(
        self,
        population_size: int,
        input_size: int,
        start_x: float,
        start_y: float,
        start_angle: float,
        elite_fraction: float = 0.2,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.08,
        hidden_sizes: tuple[int, ...] = (16, 16),
    ) -> None:
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.hidden_sizes = hidden_sizes
        self.start_x = start_x
        self.start_y = start_y
        self.start_angle = start_angle

        self.genomes: List[Genome] = []
        for _ in range(population_size):
            net = NeuralNetwork(input_size=input_size, hidden_sizes=hidden_sizes)
            car = Car(x=self.start_x, y=self.start_y, angle=self.start_angle)
            self.genomes.append(Genome(network=net, car=car))

        self.best_fitness_ever = -np.inf
        self.best_weights_ever: List[np.ndarray] | None = None

    def reset_cars(self, start_x: float, start_y: float, start_angle: float) -> None:
        self.start_x = start_x
        self.start_y = start_y
        self.start_angle = start_angle
        for genome in self.genomes:
            genome.car.reset(start_x, start_y, start_angle)
            genome.fitness = 0.0

    def step_population(self, track: Track) -> None:
        for genome in self.genomes:
            if genome.car.alive:
                state = genome.car.get_network_input(track)
                steering, throttle = genome.network.predict_action(state)
                genome.car.step(steering, throttle, track)
                genome.fitness = genome.car.fitness

    def living_count(self) -> int:
        return sum(1 for g in self.genomes if g.car.alive)

    def best_genome(self) -> Genome:
        return max(self.genomes, key=lambda g: g.fitness)

    def evolve(self) -> dict:
        """Run one GA epoch and return summary stats."""
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)
        elite_count = max(2, int(self.population_size * self.elite_fraction))
        elites = self.genomes[:elite_count]

        best = elites[0]
        if best.fitness > self.best_fitness_ever:
            self.best_fitness_ever = best.fitness
            self.best_weights_ever = best.network.get_weights()

        new_genomes: List[Genome] = []

        # Carry over elites directly (elitism).
        for elite in elites:
            cloned_net = elite.network.clone()
            new_genomes.append(
                Genome(
                    network=cloned_net,
                    car=Car(x=self.start_x, y=self.start_y, angle=self.start_angle),
                )
            )

        # Fill remaining slots with crossover + mutation children.
        while len(new_genomes) < self.population_size:
            p1, p2 = np.random.choice(elites, size=2, replace=True)
            child_weights = self.crossover(p1.network.get_weights(), p2.network.get_weights())
            child_weights = self.mutate(child_weights)
            child_net = NeuralNetwork(input_size=p1.network.input_size, hidden_sizes=self.hidden_sizes)
            child_net.set_weights(child_weights)
            new_genomes.append(
                Genome(
                    network=child_net,
                    car=Car(x=self.start_x, y=self.start_y, angle=self.start_angle),
                )
            )

        self.genomes = new_genomes
        return {
            "best_fitness": float(best.fitness),
            "avg_fitness": float(np.mean([g.fitness for g in elites])),
            "elite_count": elite_count,
        }

    def crossover(self, w1: List[np.ndarray], w2: List[np.ndarray]) -> List[np.ndarray]:
        """Uniform crossover across all weight tensors."""
        child = []
        for a, b in zip(w1, w2):
            mask = np.random.rand(*a.shape) < 0.5
            child.append(np.where(mask, a, b))
        return child

    def mutate(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Gaussian mutation on random subset of params."""
        mutated: List[np.ndarray] = []
        for w in weights:
            mask = np.random.rand(*w.shape) < self.mutation_rate
            noise = np.random.normal(0.0, self.mutation_scale, size=w.shape)
            mutated.append(w + mask * noise)
        return mutated

    def save_best(self, path: str) -> bool:
        if self.best_weights_ever is None:
            return False
        best = NeuralNetwork(input_size=self.genomes[0].network.input_size, hidden_sizes=self.hidden_sizes)
        best.set_weights(self.best_weights_ever)
        best.save_weights(path)
        return True

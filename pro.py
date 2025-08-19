import random
import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import copy
from collections import defaultdict


@dataclass
class Barang:
    id: str
    berat: float  # kg
    dimensi: float  # m³
    kota_tujuan: str
    profit_per_kg: float

    @property
    def total_profit(self) -> float:
        return self.berat * self.profit_per_kg

    @property
    def value_density(self) -> float:
        """Profit per unit space (weight + volume)"""
        return self.total_profit / (self.berat + self.dimensi * 100)


@dataclass
class Truk:
    id: int
    kapasitas_berat: float  # kg
    kapasitas_dimensi: float  # m³
    konsumsi_bbm: float  # liter/km
    kota_asal: str = "Jakarta"


class Kromosom:
    def __init__(self, jumlah_barang: int, jumlah_truk: int):
        # Inisialisasi dengan strategi yang lebih baik
        self.barang_ke_truk = self._smart_initialization(jumlah_barang, jumlah_truk)
        self.urutan_kota_per_truk = [[] for _ in range(jumlah_truk)]
        self.fitness = 0.0
        self.total_profit = 0.0
        self.total_cost = 0.0
        self.penalti = 0.0
        self.penalti_barang_tidak_dimuat = 0.0
        self.is_valid = True

    def _smart_initialization(self, jumlah_barang: int, jumlah_truk: int) -> List[int]:
        """Inisialisasi yang lebih cerdas dengan probabilitas tidak memuat yang lebih kecil"""
        assignments = []
        for _ in range(jumlah_barang):
            if random.random() < 0.85:  # 85% chance dimuat
                assignments.append(random.randint(0, jumlah_truk - 1))
            else:  # 15% chance tidak dimuat
                assignments.append(-1)
        return assignments


class ImprovedACOSolver:
    def __init__(self, jarak_matrix: Dict[Tuple[str, str], float],
                 alpha=1.0, beta=3.0, rho=0.6, n_ants=15, n_iterations=30):
        self.jarak_matrix = jarak_matrix
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.pheromone = {}
        self.best_distances = {}  # Cache untuk hasil terbaik

    def solve_tsp(self, kota_list: List[str], kota_asal: str = "Jakarta") -> Tuple[List[str], float]:
        if not kota_list:
            return [], 0.0

        if len(kota_list) == 1:
            jarak = self.jarak_matrix.get((kota_asal, kota_list[0]), 0) * 2
            return kota_list, jarak

        # Cache checking
        cache_key = (kota_asal, tuple(sorted(kota_list)))
        if cache_key in self.best_distances:
            return self.best_distances[cache_key]

        # Nearest neighbor heuristic untuk inisialisasi
        nn_route, nn_distance = self._nearest_neighbor(kota_list, kota_asal)

        all_cities = [kota_asal] + kota_list
        n_cities = len(all_cities)

        # Inisialisasi pheromone dengan hasil nearest neighbor
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    city1, city2 = all_cities[i], all_cities[j]
                    self.pheromone[(city1, city2)] = 1.0 / nn_distance if nn_distance > 0 else 1.0

        best_route = nn_route
        best_distance = nn_distance

        for iteration in range(self.n_iterations):
            routes = []
            distances = []

            for ant in range(self.n_ants):
                route = [kota_asal]
                unvisited = kota_list.copy()
                current_city = kota_asal
                total_distance = 0

                while unvisited:
                    next_city = self._select_next_city(current_city, unvisited)
                    route.append(next_city)
                    total_distance += self.jarak_matrix.get((current_city, next_city), 1000)
                    current_city = next_city
                    unvisited.remove(next_city)

                # Kembali ke asal
                total_distance += self.jarak_matrix.get((current_city, kota_asal), 1000)

                routes.append(route[1:])
                distances.append(total_distance)

                if total_distance < best_distance:
                    best_distance = total_distance
                    best_route = route[1:]

            # Update pheromone dengan elitist strategy
            self._update_pheromone(routes, distances, best_route, best_distance, kota_asal)

        # Simpan ke cache
        self.best_distances[cache_key] = (best_route, best_distance)
        return best_route, best_distance

    def _nearest_neighbor(self, kota_list: List[str], kota_asal: str) -> Tuple[List[str], float]:
        """Nearest neighbor heuristic untuk inisialisasi yang baik"""
        route = []
        unvisited = kota_list.copy()
        current_city = kota_asal
        total_distance = 0

        while unvisited:
            nearest_city = min(unvisited,
                               key=lambda city: self.jarak_matrix.get((current_city, city), 1000))
            route.append(nearest_city)
            total_distance += self.jarak_matrix.get((current_city, nearest_city), 1000)
            current_city = nearest_city
            unvisited.remove(nearest_city)

        total_distance += self.jarak_matrix.get((current_city, kota_asal), 1000)
        return route, total_distance

    def _select_next_city(self, current_city: str, unvisited: List[str]) -> str:
        """Pemilihan kota berikutnya dengan roulette wheel yang diperbaiki"""
        probabilities = []
        for city in unvisited:
            pheromone_val = self.pheromone.get((current_city, city), 1.0)
            distance = self.jarak_matrix.get((current_city, city), 1000)
            heuristic = 1.0 / distance if distance > 0 else 1.0
            prob = (pheromone_val ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(prob)

        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
            # Tambahkan sedikit exploitasi untuk kota terdekat
            if random.random() < 0.1:  # 10% chance pilih yang terdekat
                nearest_idx = min(range(len(unvisited)),
                                  key=lambda i: self.jarak_matrix.get((current_city, unvisited[i]), 1000))
                return unvisited[nearest_idx]
            else:
                return np.random.choice(unvisited, p=probabilities)
        else:
            return random.choice(unvisited)

    def _update_pheromone(self, routes, distances, best_route, best_distance, kota_asal):
        """Update pheromone dengan elitist strategy"""
        # Evaporation
        for key in self.pheromone:
            self.pheromone[key] *= (1 - self.rho)

        # Reinforcement dari semua ant
        for i, route in enumerate(routes):
            full_route = [kota_asal] + route + [kota_asal]
            pheromone_deposit = 1.0 / distances[i] if distances[i] > 0 else 1.0

            for j in range(len(full_route) - 1):
                city1, city2 = full_route[j], full_route[j + 1]
                if (city1, city2) in self.pheromone:
                    self.pheromone[(city1, city2)] += pheromone_deposit

        # Extra reinforcement untuk best route (elitist)
        if best_route:
            full_best_route = [kota_asal] + best_route + [kota_asal]
            elite_deposit = 2.0 / best_distance if best_distance > 0 else 2.0

            for j in range(len(full_best_route) - 1):
                city1, city2 = full_best_route[j], full_best_route[j + 1]
                if (city1, city2) in self.pheromone:
                    self.pheromone[(city1, city2)] += elite_deposit


class OptimizedGAACOOptimizer:
    def __init__(self, barang_list: List[Barang], truk_list: List[Truk],
                 jarak_matrix: Dict[Tuple[str, str], float], harga_bbm: float = 15000):
        self.barang_list = barang_list
        self.truk_list = truk_list
        self.jarak_matrix = jarak_matrix
        self.harga_bbm = harga_bbm
        self.jumlah_barang = len(barang_list)
        self.jumlah_truk = len(truk_list)
        self.aco_solver = ImprovedACOSolver(jarak_matrix)

        # Sort barang berdasarkan value density untuk heuristic
        self.sorted_barang_by_value = sorted(
            enumerate(barang_list),
            key=lambda x: x[1].value_density,
            reverse=True
        )

    def create_smart_kromosom(self) -> Kromosom:
        """Membuat kromosom dengan heuristic yang lebih baik"""
        kromosom = Kromosom(self.jumlah_barang, self.jumlah_truk)

        # Reset assignment
        kromosom.barang_ke_truk = [-1] * self.jumlah_barang

        # Track kapasitas truk
        truk_berat = [0.0] * self.jumlah_truk
        truk_dimensi = [0.0] * self.jumlah_truk

        # Greedy assignment berdasarkan value density
        for barang_idx, barang in self.sorted_barang_by_value:
            best_truk = -1
            best_efficiency = -1

            for truk_idx, truk in enumerate(self.truk_list):
                # Cek apakah masih muat
                if (truk_berat[truk_idx] + barang.berat <= truk.kapasitas_berat and
                        truk_dimensi[truk_idx] + barang.dimensi <= truk.kapasitas_dimensi):

                    # Hitung efisiensi (profit vs space utilization)
                    remaining_weight = truk.kapasitas_berat - truk_berat[truk_idx]
                    remaining_volume = truk.kapasitas_dimensi - truk_dimensi[truk_idx]

                    efficiency = barang.value_density * (1 + remaining_weight + remaining_volume) / 1000

                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_truk = truk_idx

            if best_truk >= 0:
                kromosom.barang_ke_truk[barang_idx] = best_truk
                truk_berat[best_truk] += barang.berat
                truk_dimensi[best_truk] += barang.dimensi

        return kromosom

    def create_random_kromosom(self) -> Kromosom:
        return Kromosom(self.jumlah_barang, self.jumlah_truk)

    def evaluate_kromosom(self, kromosom: Kromosom) -> None:
        total_profit = 0.0
        total_cost = 0.0
        penalti = 0.0
        penalti_barang_tidak_dimuat = 0.0

        kromosom.urutan_kota_per_truk = [[] for _ in range(self.jumlah_truk)]

        # Kelompokkan barang per truk
        truk_loads = {i: {'berat': 0, 'dimensi': 0, 'kota': set(), 'barang': []}
                      for i in range(self.jumlah_truk)}

        for barang_idx, truk_idx in enumerate(kromosom.barang_ke_truk):
            barang = self.barang_list[barang_idx]

            if truk_idx >= 0:
                truk_loads[truk_idx]['berat'] += barang.berat
                truk_loads[truk_idx]['dimensi'] += barang.dimensi
                truk_loads[truk_idx]['kota'].add(barang.kota_tujuan)
                truk_loads[truk_idx]['barang'].append(barang)
                total_profit += barang.total_profit
            else:
                # Penalti yang lebih adaptif berdasarkan value density
                penalti_barang_tidak_dimuat += barang.total_profit * 1.2

        # Evaluasi setiap truk
        for truk_idx, truk in enumerate(self.truk_list):
            load = truk_loads[truk_idx]

            # Penalti overload yang lebih gradual
            if load['berat'] > truk.kapasitas_berat:
                excess_weight = load['berat'] - truk.kapasitas_berat
                penalti += excess_weight * 50000  # Penalti per kg excess

            if load['dimensi'] > truk.kapasitas_dimensi:
                excess_volume = load['dimensi'] - truk.kapasitas_dimensi
                penalti += excess_volume * 100000  # Penalti per m³ excess

            # Optimasi rute jika ada barang
            if load['kota']:
                kota_list = list(load['kota'])
                best_route, total_distance = self.aco_solver.solve_tsp(kota_list, truk.kota_asal)
                kromosom.urutan_kota_per_truk[truk_idx] = best_route

                # Hitung biaya BBM
                biaya_bbm = total_distance * truk.konsumsi_bbm * self.harga_bbm / 100
                total_cost += biaya_bbm

        kromosom.total_profit = total_profit
        kromosom.total_cost = total_cost
        kromosom.penalti = penalti
        kromosom.penalti_barang_tidak_dimuat = penalti_barang_tidak_dimuat
        kromosom.is_valid = penalti == 0

        # Fitness function yang lebih seimbang
        kromosom.fitness = total_profit - total_cost - penalti - penalti_barang_tidak_dimuat

    def adaptive_tournament_selection(self, populasi: List[Kromosom], generation: int, max_gen: int) -> Kromosom:
        """Tournament selection yang adaptif"""
        # Tournament size yang adaptif
        progress = generation / max_gen
        k = max(2, min(7, int(3 + 4 * progress)))  # Mulai dari 3, naik ke 7

        tournament = random.sample(populasi, min(k, len(populasi)))

        # Bias terhadap solusi valid di awal generasi
        if generation < max_gen * 0.3:
            valid_solutions = [k for k in tournament if k.is_valid]
            if valid_solutions:
                return max(valid_solutions, key=lambda x: x.fitness)

        return max(tournament, key=lambda x: x.fitness)

    def smart_crossover(self, parent1: Kromosom, parent2: Kromosom) -> Tuple[Kromosom, Kromosom]:
        """Crossover yang lebih cerdas dengan multiple strategies"""
        child1 = Kromosom(self.jumlah_barang, self.jumlah_truk)
        child2 = Kromosom(self.jumlah_barang, self.jumlah_truk)

        # Pilih strategi crossover secara random
        strategy = random.choice(['single_point', 'two_point', 'uniform', 'value_based'])

        if strategy == 'single_point':
            crossover_point = random.randint(1, self.jumlah_barang - 1)
            child1.barang_ke_truk = (parent1.barang_ke_truk[:crossover_point] +
                                     parent2.barang_ke_truk[crossover_point:])
            child2.barang_ke_truk = (parent2.barang_ke_truk[:crossover_point] +
                                     parent1.barang_ke_truk[crossover_point:])

        elif strategy == 'two_point':
            p1, p2 = sorted(random.sample(range(1, self.jumlah_barang), 2))
            child1.barang_ke_truk = (parent1.barang_ke_truk[:p1] +
                                     parent2.barang_ke_truk[p1:p2] +
                                     parent1.barang_ke_truk[p2:])
            child2.barang_ke_truk = (parent2.barang_ke_truk[:p1] +
                                     parent1.barang_ke_truk[p1:p2] +
                                     parent2.barang_ke_truk[p2:])

        elif strategy == 'uniform':
            child1.barang_ke_truk = []
            child2.barang_ke_truk = []
            for i in range(self.jumlah_barang):
                if random.random() < 0.5:
                    child1.barang_ke_truk.append(parent1.barang_ke_truk[i])
                    child2.barang_ke_truk.append(parent2.barang_ke_truk[i])
                else:
                    child1.barang_ke_truk.append(parent2.barang_ke_truk[i])
                    child2.barang_ke_truk.append(parent1.barang_ke_truk[i])

        else:  # value_based
            # Pilih parent berdasarkan value density barang
            child1.barang_ke_truk = []
            child2.barang_ke_truk = []
            for i in range(self.jumlah_barang):
                barang = self.barang_list[i]
                # Barang dengan value density tinggi lebih cenderung dari parent yang lebih baik
                if parent1.fitness > parent2.fitness:
                    prob = 0.7 if barang.value_density > 50 else 0.5
                else:
                    prob = 0.3 if barang.value_density > 50 else 0.5

                if random.random() < prob:
                    child1.barang_ke_truk.append(parent1.barang_ke_truk[i])
                    child2.barang_ke_truk.append(parent2.barang_ke_truk[i])
                else:
                    child1.barang_ke_truk.append(parent2.barang_ke_truk[i])
                    child2.barang_ke_truk.append(parent1.barang_ke_truk[i])

        self.smart_repair(child1)
        self.smart_repair(child2)

        return child1, child2

    def adaptive_mutate(self, kromosom: Kromosom, mutation_rate: float, generation: int, max_gen: int) -> None:
        """Mutasi yang adaptif berdasarkan generasi dan kondisi kromosom"""
        progress = generation / max_gen

        # Mutation rate yang adaptif
        if not kromosom.is_valid:
            effective_rate = mutation_rate * 2  # Mutasi lebih agresif untuk solusi invalid
        else:
            effective_rate = mutation_rate * (1.5 - progress)  # Berkurang seiring waktu

        for i in range(self.jumlah_barang):
            if random.random() < effective_rate:
                barang = self.barang_list[i]
                current_truk = kromosom.barang_ke_truk[i]

                # Strategi mutasi yang berbeda berdasarkan value density
                if barang.value_density > 50:  # Barang bernilai tinggi
                    # Lebih konservatif, coba pindah ke truk yang lebih baik
                    candidates = []
                    for truk_idx in range(self.jumlah_truk):
                        if truk_idx != current_truk:
                            candidates.append(truk_idx)
                    if candidates and random.random() < 0.8:
                        kromosom.barang_ke_truk[i] = random.choice(candidates)
                    elif random.random() < 0.1:
                        kromosom.barang_ke_truk[i] = -1
                else:  # Barang bernilai rendah
                    # Lebih berani untuk tidak dimuat atau pindah random
                    if random.random() < 0.3:
                        kromosom.barang_ke_truk[i] = -1
                    else:
                        kromosom.barang_ke_truk[i] = random.randint(0, self.jumlah_truk - 1)

    def smart_repair(self, kromosom: Kromosom) -> None:
        """Repair yang lebih cerdas dengan multiple strategies"""
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            truk_loads = self._calculate_loads(kromosom)
            overloaded_trucks = []

            for truk_idx, truk in enumerate(self.truk_list):
                load = truk_loads[truk_idx]
                if (load['berat'] > truk.kapasitas_berat or
                        load['dimensi'] > truk.kapasitas_dimensi):
                    overloaded_trucks.append(truk_idx)

            if not overloaded_trucks:
                break

            # Repair strategy berdasarkan iterasi
            if iteration < 2:
                self._repair_by_moving(kromosom, truk_loads, overloaded_trucks)
            else:
                self._repair_by_unloading(kromosom, truk_loads, overloaded_trucks)

            iteration += 1

    def _calculate_loads(self, kromosom: Kromosom) -> Dict:
        """Hitung load untuk setiap truk"""
        truk_loads = {i: {'berat': 0, 'dimensi': 0, 'barang_idx': []}
                      for i in range(self.jumlah_truk)}

        for barang_idx, truk_idx in enumerate(kromosom.barang_ke_truk):
            if truk_idx >= 0:
                barang = self.barang_list[barang_idx]
                truk_loads[truk_idx]['berat'] += barang.berat
                truk_loads[truk_idx]['dimensi'] += barang.dimensi
                truk_loads[truk_idx]['barang_idx'].append(barang_idx)

        return truk_loads

    def _repair_by_moving(self, kromosom: Kromosom, truk_loads: Dict, overloaded_trucks: List[int]) -> None:
        """Repair dengan memindahkan barang ke truk lain"""
        for truk_idx in overloaded_trucks:
            load = truk_loads[truk_idx]
            truk = self.truk_list[truk_idx]

            # Sort barang berdasarkan value density (pindahkan yang terburuk dulu)
            sorted_barang = sorted(load['barang_idx'],
                                   key=lambda idx: self.barang_list[idx].value_density)

            for barang_idx in sorted_barang:
                if (load['berat'] <= truk.kapasitas_berat and
                        load['dimensi'] <= truk.kapasitas_dimensi):
                    break

                barang = self.barang_list[barang_idx]
                moved = False

                # Coba pindah ke truk lain
                for alt_truk_idx in range(self.jumlah_truk):
                    if alt_truk_idx != truk_idx:
                        alt_load = truk_loads[alt_truk_idx]
                        alt_truk = self.truk_list[alt_truk_idx]

                        if (alt_load['berat'] + barang.berat <= alt_truk.kapasitas_berat and
                                alt_load['dimensi'] + barang.dimensi <= alt_truk.kapasitas_dimensi):
                            # Pindahkan
                            kromosom.barang_ke_truk[barang_idx] = alt_truk_idx

                            # Update loads
                            load['berat'] -= barang.berat
                            load['dimensi'] -= barang.dimensi
                            load['barang_idx'].remove(barang_idx)

                            alt_load['berat'] += barang.berat
                            alt_load['dimensi'] += barang.dimensi
                            alt_load['barang_idx'].append(barang_idx)

                            moved = True
                            break

                if not moved:
                    # Tidak bisa dipindah, unload
                    kromosom.barang_ke_truk[barang_idx] = -1
                    load['berat'] -= barang.berat
                    load['dimensi'] -= barang.dimensi
                    load['barang_idx'].remove(barang_idx)

    def _repair_by_unloading(self, kromosom: Kromosom, truk_loads: Dict, overloaded_trucks: List[int]) -> None:
        """Repair dengan unload barang yang value density rendah"""
        for truk_idx in overloaded_trucks:
            load = truk_loads[truk_idx]
            truk = self.truk_list[truk_idx]

            # Sort dan unload barang dengan value density terendah
            sorted_barang = sorted(load['barang_idx'],
                                   key=lambda idx: self.barang_list[idx].value_density)

            for barang_idx in sorted_barang:
                if (load['berat'] <= truk.kapasitas_berat and
                        load['dimensi'] <= truk.kapasitas_dimensi):
                    break

                barang = self.barang_list[barang_idx]
                kromosom.barang_ke_truk[barang_idx] = -1
                load['berat'] -= barang.berat
                load['dimensi'] -= barang.dimensi
                load['barang_idx'].remove(barang_idx)

    def optimize(self, population_size: int = 60, generations: int = 150,
                 mutation_rate: float = 0.2, elite_size: int = 12) -> Kromosom:
        """Optimasi dengan strategi yang diperbaiki"""

        # Inisialisasi populasi dengan mix smart dan random
        populasi = []
        smart_ratio = 0.3  # 30% smart initialization
        smart_count = int(population_size * smart_ratio)

        for i in range(smart_count):
            populasi.append(self.create_smart_kromosom())

        for i in range(population_size - smart_count):
            populasi.append(self.create_random_kromosom())

        # Evaluasi populasi awal
        for kromosom in populasi:
            self.evaluate_kromosom(kromosom)

        best_fitness_history = []
        stagnation_counter = 0
        best_fitness_ever = float('-inf')

        for generation in range(generations):
            # Sort populasi berdasarkan fitness
            populasi.sort(key=lambda x: x.fitness, reverse=True)

            current_best = populasi[0].fitness
            best_fitness_history.append(current_best)

            # Track stagnation
            if current_best > best_fitness_ever:
                best_fitness_ever = current_best
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            print(f"Gen {generation + 1}: Best={populasi[0].fitness:.0f}, "
                  f"Valid={sum(1 for k in populasi if k.is_valid)}/{population_size}, "
                  f"Profit={populasi[0].total_profit:.0f}, Cost={populasi[0].total_cost:.0f}")

            # Diversity injection jika stagnasi
            if stagnation_counter > 20:
                print(f"Diversity injection at generation {generation + 1}")
                # Replace 20% worst dengan random baru
                replace_count = population_size // 5
                for i in range(replace_count):
                    idx = population_size - 1 - i
                    if random.random() < 0.5:
                        populasi[idx] = self.create_smart_kromosom()
                    else:
                        populasi[idx] = self.create_random_kromosom()
                    self.evaluate_kromosom(populasi[idx])
                stagnation_counter = 0

            # Elitism
            new_population = populasi[:elite_size].copy()

            # Generate offspring
            while len(new_population) < population_size:
                parent1 = self.adaptive_tournament_selection(populasi, generation, generations)
                parent2 = self.adaptive_tournament_selection(populasi, generation, generations)

                # Crossover
                if random.random() < 0.8:  # Crossover probability
                    child1, child2 = self.smart_crossover(parent1, parent2)
                else:
                    child1 = copy.deepcopy(parent1)
                    child2 = copy.deepcopy(parent2)

                # Mutation
                self.adaptive_mutate(child1, mutation_rate, generation, generations)
                self.adaptive_mutate(child2, mutation_rate, generation, generations)

                # Evaluate children
                self.evaluate_kromosom(child1)
                self.evaluate_kromosom(child2)

                new_population.extend([child1, child2])

            # Trim to exact population size if needed
            populasi = new_population[:population_size]

            # Adaptive parameter adjustment
            if generation % 25 == 0 and generation > 0:
                # Adjust mutation rate based on diversity
                valid_count = sum(1 for k in populasi if k.is_valid)
                if valid_count > population_size * 0.8:
                    mutation_rate = min(0.4, mutation_rate * 1.1)  # Increase exploration
                else:
                    mutation_rate = max(0.1, mutation_rate * 0.9)  # Decrease mutation

            # Return best solution
            populasi.sort(key=lambda x: x.fitness, reverse=True)
        return populasi[0]


    def local_search(self, kromosom: Kromosom, max_iterations: int = 50) -> None:
        """Local search untuk fine-tuning solusi terbaik"""
        current_fitness = kromosom.fitness
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Try swapping assignments
            for i in range(self.jumlah_barang):
                if improved:
                    break

                current_assignment = kromosom.barang_ke_truk[i]

                # Try all possible reassignments
                for new_assignment in range(-1, self.jumlah_truk):
                    if new_assignment != current_assignment:
                        # Make temporary change
                        kromosom.barang_ke_truk[i] = new_assignment

                        # Check if valid and evaluate
                        temp_kromosom = copy.deepcopy(kromosom)
                        self.smart_repair(temp_kromosom)
                        self.evaluate_kromosom(temp_kromosom)

                        if temp_kromosom.fitness > current_fitness:
                            kromosom.barang_ke_truk = temp_kromosom.barang_ke_truk.copy()
                            kromosom.urutan_kota_per_truk = temp_kromosom.urutan_kota_per_truk.copy()
                            kromosom.fitness = temp_kromosom.fitness
                            kromosom.total_profit = temp_kromosom.total_profit
                            kromosom.total_cost = temp_kromosom.total_cost
                            kromosom.penalti = temp_kromosom.penalti
                            kromosom.penalti_barang_tidak_dimuat = temp_kromosom.penalti_barang_tidak_dimuat
                            kromosom.is_valid = temp_kromosom.is_valid
                            current_fitness = kromosom.fitness
                            improved = True
                            break
                        else:
                            # Revert change
                            kromosom.barang_ke_truk[i] = current_assignment


    def print_detailed_solution(self, best_kromosom: Kromosom) -> None:
        """Print detailed solution information"""
        print("\n" + "=" * 80)
        print("DETAILED SOLUTION ANALYSIS")
        print("=" * 80)

        print(f"Overall Fitness: {best_kromosom.fitness:,.0f}")
        print(f"Total Profit: Rp {best_kromosom.total_profit:,.0f}")
        print(f"Total Cost: Rp {best_kromosom.total_cost:,.0f}")
        print(f"Net Profit: Rp {best_kromosom.total_profit - best_kromosom.total_cost:,.0f}")
        print(f"Penalties: Rp {best_kromosom.penalti + best_kromosom.penalti_barang_tidak_dimuat:,.0f}")
        print(f"Valid Solution: {best_kromosom.is_valid}")

        print("\n" + "-" * 60)
        print("TRUCK ASSIGNMENTS AND ROUTES")
        print("-" * 60)

        total_loaded_items = 0

        for truk_idx, truk in enumerate(self.truk_list):
            barang_in_truk = [i for i, t in enumerate(best_kromosom.barang_ke_truk) if t == truk_idx]

            if barang_in_truk:
                total_loaded_items += len(barang_in_truk)
                total_weight = sum(self.barang_list[i].berat for i in barang_in_truk)
                total_volume = sum(self.barang_list[i].dimensi for i in barang_in_truk)
                total_profit_truk = sum(self.barang_list[i].total_profit for i in barang_in_truk)

                print(f"\nTruck {truk_idx + 1}:")
                print(f"  Capacity: {total_weight:.1f}/{truk.kapasitas_berat:.1f} kg, "
                      f"{total_volume:.2f}/{truk.kapasitas_dimensi:.2f} m³")
                print(f"  Items: {len(barang_in_truk)}, Profit: Rp {total_profit_truk:,.0f}")

                if best_kromosom.urutan_kota_per_truk[truk_idx]:
                    route = best_kromosom.urutan_kota_per_truk[truk_idx]
                    print(f"  Route: {truk.kota_asal} → {' → '.join(route)} → {truk.kota_asal}")

                # Show items details
                print(f"  Items loaded:")
                for item_idx in barang_in_truk:
                    item = self.barang_list[item_idx]
                    print(f"    - {item.id}: {item.berat}kg, {item.dimensi}m³, "
                          f"{item.kota_tujuan}, Rp {item.total_profit:,.0f}")

        unloaded_items = [i for i, t in enumerate(best_kromosom.barang_ke_truk) if t == -1]

        print(f"\n" + "-" * 60)
        print(f"SUMMARY")
        print("-" * 60)
        print(f"Items loaded: {total_loaded_items}/{self.jumlah_barang}")
        print(f"Items unloaded: {len(unloaded_items)}")

        if unloaded_items:
            unloaded_profit = sum(self.barang_list[i].total_profit for i in unloaded_items)
            print(f"Lost profit from unloaded items: Rp {unloaded_profit:,.0f}")
            print(f"Unloaded items:")
            for item_idx in unloaded_items[:10]:  # Show first 10
                item = self.barang_list[item_idx]
                print(f"  - {item.id}: {item.berat}kg, {item.dimensi}m³, "
                      f"{item.kota_tujuan}, Rp {item.total_profit:,.0f}")
            if len(unloaded_items) > 10:
                print(f"  ... and {len(unloaded_items) - 10} more items")


def run_optimization_with_local_search(barang_list: List[Barang], truk_list: List[Truk],
                                       jarak_matrix: Dict[Tuple[str, str], float],
                                       harga_bbm: float = 15000) -> Kromosom:
    """Run complete optimization with local search"""

    print("Starting GA-ACO Optimization...")
    optimizer = OptimizedGAACOOptimizer(barang_list, truk_list, jarak_matrix, harga_bbm)

    # Main GA-ACO optimization
    best_solution = optimizer.optimize(
        population_size=60,
        generations=150,
        mutation_rate=0.2,
        elite_size=12
    )

    print(f"\nGA-ACO completed. Best fitness: {best_solution.fitness:,.0f}")
    print("Starting local search refinement...")

    # Apply local search
    optimizer.local_search(best_solution, max_iterations=50)

    print(f"Local search completed. Final fitness: {best_solution.fitness:,.0f}")

    # Print detailed solution
    optimizer.print_detailed_solution(best_solution)

    return best_solution


# Example usage and testing
if __name__ == "__main__":
    # Example data untuk testing

    # Sample barang data
    sample_barang = [
        Barang("B001", 100, 0.5, "Bandung", 5000),
        Barang("B002", 150, 0.8, "Surabaya", 4500),
        Barang("B003", 80, 0.3, "Yogyakarta", 6000),
        Barang("B004", 200, 1.2, "Malang", 4000),
        Barang("B005", 120, 0.6, "Solo", 5500),
        Barang("B006", 90, 0.4, "Semarang", 5200),
        Barang("B007", 160, 0.9, "Bandung", 4800),
        Barang("B008", 110, 0.5, "Surabaya", 5300),
        Barang("B009", 130, 0.7, "Yogyakarta", 5100),
        Barang("B010", 170, 1.0, "Malang", 4900),
        Barang("B011", 95, 0.4, "Solo", 5600),
        Barang("B012", 105, 0.6, "Semarang", 4700),
        Barang("B013", 140, 0.9, "Bandung", 5200),
        Barang("B014", 115, 0.5, "Surabaya", 5000),
        Barang("B015", 125, 0.7, "Yogyakarta", 5400),
        Barang("B016", 190, 1.1, "Malang", 4300),
        Barang("B017", 100, 0.5, "Solo", 4900),
        Barang("B018", 85, 0.3, "Semarang", 5800),
        Barang("B019", 145, 0.8, "Bandung", 5100),
        Barang("B020", 155, 0.9, "Surabaya", 4700),
        Barang("B021", 135, 0.7, "Yogyakarta", 5000),
        Barang("B022", 175, 1.0, "Malang", 4500),
        Barang("B023", 120, 0.6, "Solo", 5300),
        Barang("B024", 90, 0.4, "Semarang", 5600),
        Barang("B025", 150, 0.8, "Bandung", 4900),
        Barang("B026", 110, 0.5, "Surabaya", 5200),
        Barang("B027", 100, 0.6, "Yogyakarta", 5100),
        Barang("B028", 180, 1.1, "Malang", 4600),
        Barang("B029", 130, 0.7, "Solo", 5000),
        Barang("B030", 95, 0.4, "Semarang", 5500),
    ]

    # Sample truk data
    sample_truk = [
        Truk(1, 2000, 9.0, 0.15, "Jakarta"),
        Truk(2, 1500, 8.0, 0.18, "Jakarta"),
        Truk(3, 1000, 6.0, 0.12, "Jakarta"),
    ]

    # Sample jarak matrix (simplified)
    sample_jarak = {
        ("Jakarta", "Surabaya"): 800,
        ("Surabaya", "Jakarta"): 800,
        ("Jakarta", "Bandung"): 150,
        ("Bandung", "Jakarta"): 150,
        ("Jakarta", "Medan"): 1400,
        ("Medan", "Jakarta"): 1400,
        ("Jakarta", "Yogyakarta"): 560,
        ("Yogyakarta", "Jakarta"): 560,
        ("Surabaya", "Bandung"): 950,
        ("Bandung", "Surabaya"): 950,
        ("Surabaya", "Medan"): 2200,
        ("Medan", "Surabaya"): 2200,
        ("Surabaya", "Yogyakarta"): 320,
        ("Yogyakarta", "Surabaya"): 320,
        ("Bandung", "Medan"): 1550,
        ("Medan", "Bandung"): 1550,
        ("Bandung", "Yogyakarta"): 420,
        ("Yogyakarta", "Bandung"): 420,
        ("Medan", "Yogyakarta"): 1960,
        ("Yogyakarta", "Medan"): 1960,
    }

    print("Running example optimization...")
    best_solution = run_optimization_with_local_search(
        sample_barang, sample_truk, sample_jarak, harga_bbm=15000
    )

    print(f"\nFinal Solution Summary:")
    print(f"Fitness: {best_solution.fitness:,.0f}")
    print(f"Total Profit: Rp {best_solution.total_profit:,.0f}")
    print(f"Total Cost: Rp {best_solution.total_cost:,.0f}")
    print(f"Net Profit: Rp {best_solution.total_profit - best_solution.total_cost:,.0f}")
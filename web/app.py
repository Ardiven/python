from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import json
from datetime import datetime
import os
from typing import List, Dict, Tuple, Set, Optional
import threading
import time
import networkx as nx
import random
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Global variables to store data
barang_list = []
truk_list = []
network_matrix = {}
optimization_results = {}
optimization_status = {"running": False, "progress": 0, "current_generation": 0}


# Core Classes
class Barang:
    def __init__(self, id: str, berat: float, dimensi: float, kota_tujuan: str,
                 base_price: float, urgency: int):
        self.id = id
        self.berat = berat
        self.dimensi = dimensi
        self.kota_tujuan = kota_tujuan
        self.base_price = base_price
        self.urgency = urgency
        self.base_total_profit = base_price * (1 + urgency * 0.2)

    def calculate_profit(self, distance):
        # Transport cost calculation: distance * rate per km
        transport_cost = distance * 0.8  # Cost per km
        # Urgency bonus decreases with distance
        urgency_bonus = self.base_price * (self.urgency * 0.1) * max(0, 1 - distance / 1000)
        return max(0, self.base_total_profit + urgency_bonus - transport_cost)

    def to_dict(self):
        """Convert Barang object to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'berat': self.berat,
            'dimensi': self.dimensi,
            'kota_tujuan': self.kota_tujuan,
            'base_price': self.base_price,
            'urgency': self.urgency,
            'base_total_profit': self.base_total_profit
        }


class Truk:
    def __init__(self, id: int, kapasitas_berat: float, kapasitas_dimensi: float,
                 konsumsi_bbm: float, kota_asal: str):
        self.id = id
        self.kapasitas_berat = kapasitas_berat
        self.kapasitas_dimensi = kapasitas_dimensi
        self.konsumsi_bbm = konsumsi_bbm
        self.kota_asal = kota_asal

    def can_deliver_to(self, kota):
        return True  # For now, assume all trucks can deliver anywhere

    def to_dict(self):
        """Convert Truk object to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'kapasitas_berat': self.kapasitas_berat,
            'kapasitas_dimensi': self.kapasitas_dimensi,
            'konsumsi_bbm': self.konsumsi_bbm,
            'kota_asal': self.kota_asal
        }


class Kromosom:
    def __init__(self, jumlah_barang: int, jumlah_truk: int):
        self.barang_ke_truk = [-1] * jumlah_barang  # -1 means not assigned
        self.urutan_kota_per_truk = [[] for _ in range(jumlah_truk)]
        self.routing_details = [None for _ in range(jumlah_truk)]
        self.total_profit = 0.0
        self.total_cost = 0.0
        self.penalti = 0.0
        self.penalti_barang_tidak_dimuat = 0.0
        self.fitness = 0.0
        self.is_valid = False


# Enhanced ACO Solver
class EnhancedACOSolver:
    def __init__(self, jarak_matrix: Dict[Tuple[str, str], float],
                 hub_cities: List[str] = None,
                 alpha=1.0, beta=3.0, rho=0.6, n_ants=15, n_iterations=30):
        self.jarak_matrix = jarak_matrix
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.pheromone = {}
        self.best_distances = {}

        # Hub cities yang bisa menjadi transit point
        self.hub_cities = hub_cities or ["Jakarta", "Surabaya", "Medan", "Makassar"]

        # Build network graph untuk shortest path calculation
        self.network_graph = self._build_network_graph()

    def _build_network_graph(self) -> nx.Graph:
        """Build network graph from distance matrix"""
        G = nx.Graph()

        for (city1, city2), distance in self.jarak_matrix.items():
            G.add_edge(city1, city2, weight=distance)

        return G

    def _get_shortest_path_distance(self, city1: str, city2: str) -> float:
        """Get shortest path distance between two cities using network routing"""
        try:
            if city1 == city2:
                return 0.0

            # Check direct connection first
            if (city1, city2) in self.jarak_matrix:
                return self.jarak_matrix[(city1, city2)]
            elif (city2, city1) in self.jarak_matrix:
                return self.jarak_matrix[(city2, city1)]

            # Use shortest path through network
            if self.network_graph.has_node(city1) and self.network_graph.has_node(city2):
                try:
                    return nx.shortest_path_length(self.network_graph, city1, city2, weight='weight')
                except nx.NetworkXNoPath:
                    # No path exists, use large penalty
                    return 10000.0
            else:
                return 10000.0  # Very large distance for unconnected cities

        except Exception:
            return 10000.0

    def _get_shortest_path_route(self, city1: str, city2: str) -> List[str]:
        """Get actual route (intermediate cities) for shortest path"""
        try:
            if city1 == city2:
                return []

            # Check direct connection
            if ((city1, city2) in self.jarak_matrix or
                    (city2, city1) in self.jarak_matrix):
                return []  # Direct connection, no intermediate cities

            # Find shortest path through network
            if self.network_graph.has_node(city1) and self.network_graph.has_node(city2):
                try:
                    path = nx.shortest_path(self.network_graph, city1, city2, weight='weight')
                    # Return intermediate cities (exclude start and end)
                    return path[1:-1] if len(path) > 2 else []
                except nx.NetworkXNoPath:
                    return []

            return []

        except Exception:
            return []

    def solve_tsp_with_routing(self, kota_list: List[str], kota_asal: str = "Jakarta") -> Tuple[List[str], float, Dict]:
        """
        Enhanced TSP solver that considers network routing constraints
        Returns: (route, total_distance, detailed_routing_info)
        """
        if not kota_list:
            return [], 0.0, {}

        if len(kota_list) == 1:
            distance = self._get_shortest_path_distance(kota_asal, kota_list[0])
            return_distance = self._get_shortest_path_distance(kota_list[0], kota_asal)
            total_distance = distance + return_distance

            # Get detailed routing
            route_to = self._get_shortest_path_route(kota_asal, kota_list[0])
            route_back = self._get_shortest_path_route(kota_list[0], kota_asal)

            routing_info = {
                'segments': [
                    {
                        'from': kota_asal,
                        'to': kota_list[0],
                        'distance': distance,
                        'intermediate_cities': route_to
                    },
                    {
                        'from': kota_list[0],
                        'to': kota_asal,
                        'distance': return_distance,
                        'intermediate_cities': route_back
                    }
                ],
                'total_distance': total_distance,
                'unique_intermediate_cities': list(set(route_to + route_back))
            }

            return kota_list, total_distance, routing_info

        # Use ACO for multiple cities
        best_route, best_distance, routing_info = self._aco_with_routing(kota_list, kota_asal)

        return best_route, best_distance, routing_info

    def _aco_with_routing(self, kota_list: List[str], kota_asal: str) -> Tuple[List[str], float, Dict]:
        """ACO implementation that considers network routing"""

        # Initialize pheromones
        all_cities = [kota_asal] + kota_list
        for i, city1 in enumerate(all_cities):
            for j, city2 in enumerate(all_cities):
                if i != j:
                    distance = self._get_shortest_path_distance(city1, city2)
                    self.pheromone[(city1, city2)] = 1.0 / distance if distance > 0 else 1.0

        # Get nearest neighbor solution as baseline
        nn_route, nn_distance, nn_routing = self._nearest_neighbor_with_routing(kota_list, kota_asal)

        best_route = nn_route
        best_distance = nn_distance
        best_routing_info = nn_routing

        # ACO iterations
        for iteration in range(self.n_iterations):
            routes = []
            distances = []
            routing_infos = []

            for ant in range(self.n_ants):
                route = [kota_asal]
                unvisited = kota_list.copy()
                current_city = kota_asal
                total_distance = 0
                segments = []

                # Build route
                while unvisited:
                    next_city = self._select_next_city_routing(current_city, unvisited)

                    segment_distance = self._get_shortest_path_distance(current_city, next_city)
                    intermediate_cities = self._get_shortest_path_route(current_city, next_city)

                    segments.append({
                        'from': current_city,
                        'to': next_city,
                        'distance': segment_distance,
                        'intermediate_cities': intermediate_cities
                    })

                    route.append(next_city)
                    total_distance += segment_distance
                    current_city = next_city
                    unvisited.remove(next_city)

                # Return to start
                return_distance = self._get_shortest_path_distance(current_city, kota_asal)
                return_intermediate = self._get_shortest_path_route(current_city, kota_asal)

                segments.append({
                    'from': current_city,
                    'to': kota_asal,
                    'distance': return_distance,
                    'intermediate_cities': return_intermediate
                })

                total_distance += return_distance

                # Create routing info
                all_intermediate = []
                for segment in segments:
                    all_intermediate.extend(segment['intermediate_cities'])

                routing_info = {
                    'segments': segments,
                    'total_distance': total_distance,
                    'unique_intermediate_cities': list(set(all_intermediate))
                }

                routes.append(route[1:])  # Exclude starting city
                distances.append(total_distance)
                routing_infos.append(routing_info)

                # Update best solution
                if total_distance < best_distance:
                    best_distance = total_distance
                    best_route = route[1:]
                    best_routing_info = routing_info

            # Update pheromones
            self._update_pheromone_routing(routes, distances, best_route, best_distance, kota_asal)

        return best_route, best_distance, best_routing_info

    def _nearest_neighbor_with_routing(self, kota_list: List[str], kota_asal: str) -> Tuple[List[str], float, Dict]:
        """Nearest neighbor heuristic with routing consideration"""
        route = []
        unvisited = kota_list.copy()
        current_city = kota_asal
        total_distance = 0
        segments = []

        while unvisited:
            nearest_city = min(unvisited,
                               key=lambda city: self._get_shortest_path_distance(current_city, city))

            distance = self._get_shortest_path_distance(current_city, nearest_city)
            intermediate_cities = self._get_shortest_path_route(current_city, nearest_city)

            segments.append({
                'from': current_city,
                'to': nearest_city,
                'distance': distance,
                'intermediate_cities': intermediate_cities
            })

            route.append(nearest_city)
            total_distance += distance
            current_city = nearest_city
            unvisited.remove(nearest_city)

        # Return to start
        return_distance = self._get_shortest_path_distance(current_city, kota_asal)
        return_intermediate = self._get_shortest_path_route(current_city, kota_asal)

        segments.append({
            'from': current_city,
            'to': kota_asal,
            'distance': return_distance,
            'intermediate_cities': return_intermediate
        })

        total_distance += return_distance

        # Create routing info
        all_intermediate = []
        for segment in segments:
            all_intermediate.extend(segment['intermediate_cities'])

        routing_info = {
            'segments': segments,
            'total_distance': total_distance,
            'unique_intermediate_cities': list(set(all_intermediate))
        }

        return route, total_distance, routing_info

    def _select_next_city_routing(self, current_city: str, unvisited: List[str]) -> str:
        """Select next city considering routing constraints"""
        probabilities = []

        for city in unvisited:
            pheromone_val = self.pheromone.get((current_city, city), 1.0)
            distance = self._get_shortest_path_distance(current_city, city)

            # Penalize very long routes (likely through many hubs)
            if distance > 2000:  # Adjust threshold as needed
                distance *= 1.5

            heuristic = 1.0 / distance if distance > 0 else 1.0
            prob = (pheromone_val ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(prob)

        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]

            # Sometimes choose greedily
            if random.random() < 0.1:
                nearest_idx = min(range(len(unvisited)),
                                  key=lambda i: self._get_shortest_path_distance(current_city, unvisited[i]))
                return unvisited[nearest_idx]
            else:
                return np.random.choice(unvisited, p=probabilities)
        else:
            return random.choice(unvisited)

    def _update_pheromone_routing(self, routes, distances, best_route, best_distance, kota_asal):
        """Update pheromone considering routing"""
        # Evaporation
        for key in self.pheromone:
            self.pheromone[key] *= (1 - self.rho)

        # Deposit pheromones for all routes
        for i, route in enumerate(routes):
            full_route = [kota_asal] + route + [kota_asal]
            pheromone_deposit = 1.0 / distances[i] if distances[i] > 0 else 1.0

            for j in range(len(full_route) - 1):
                city1, city2 = full_route[j], full_route[j + 1]
                if (city1, city2) in self.pheromone:
                    self.pheromone[(city1, city2)] += pheromone_deposit

        # Extra deposit for best route
        if best_route:
            full_best_route = [kota_asal] + best_route + [kota_asal]
            elite_deposit = 2.0 / best_distance if best_distance > 0 else 2.0

            for j in range(len(full_best_route) - 1):
                city1, city2 = full_best_route[j], full_best_route[j + 1]
                if (city1, city2) in self.pheromone:
                    self.pheromone[(city1, city2)] += elite_deposit


# Optimized GA ACO Optimizer
class OptimizedGAACOOptimizer:
    def __init__(self, barang_list: List[Barang], truk_list: List[Truk],
                 jarak_matrix: Dict[Tuple[str, str], float], harga_bbm: float = 15000,
                 use_network_routing: bool = True, hub_cities: List[str] = None):
        self.barang_list = barang_list
        self.truk_list = truk_list
        self.jarak_matrix = jarak_matrix
        self.harga_bbm = harga_bbm
        self.jumlah_barang = len(barang_list)
        self.jumlah_truk = len(truk_list)

        # Choose ACO solver based on routing preference
        if use_network_routing:
            self.aco_solver = EnhancedACOSolver(
                jarak_matrix,
                hub_cities=hub_cities or ["Jakarta", "Surabaya", "Medan", "Makassar"]
            )
        else:
            # Fallback to simple distance calculation
            self.aco_solver = None

        self.use_network_routing = use_network_routing

        # Pre-calculate distances considering network routing
        self.barang_distances = {}
        for barang in barang_list:
            for truk in truk_list:
                key = (barang.id, truk.kota_asal, barang.kota_tujuan)
                if use_network_routing and self.aco_solver:
                    distance = self.aco_solver._get_shortest_path_distance(
                        truk.kota_asal, barang.kota_tujuan
                    )
                else:
                    distance = jarak_matrix.get((truk.kota_asal, barang.kota_tujuan),
                                                jarak_matrix.get((barang.kota_tujuan, truk.kota_asal), 1000))
                self.barang_distances[key] = distance

    def optimize(self, population_size=50, generations=100, callback=None):
        """Run the optimization algorithm"""
        if not self.barang_list or not self.truk_list:
            raise ValueError("Barang list and truk list cannot be empty")

        # Initialize population
        population = []
        for _ in range(population_size):
            kromosom = self._create_random_kromosom()
            self.evaluate_kromosom(kromosom)
            population.append(kromosom)

        best_kromosom = max(population, key=lambda k: k.fitness)

        # Evolution
        for generation in range(generations):
            if callback:
                callback(generation + 1, generations)

            # Selection and reproduction
            new_population = []

            # Elitism - keep best 20%
            elite_size = max(1, population_size // 5)
            population.sort(key=lambda k: k.fitness, reverse=True)
            new_population.extend(population[:elite_size])

            # Generate rest of population
            while len(new_population) < population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                child = self._crossover(parent1, parent2)
                child = self._mutate(child)

                self.evaluate_kromosom(child)
                new_population.append(child)

            population = new_population

            # Update best solution
            current_best = max(population, key=lambda k: k.fitness)
            if current_best.fitness > best_kromosom.fitness:
                best_kromosom = current_best

        return best_kromosom

    def _create_random_kromosom(self) -> Kromosom:
        """Create a random chromosome"""
        kromosom = Kromosom(self.jumlah_barang, self.jumlah_truk)

        for i in range(self.jumlah_barang):
            # 80% chance to assign to a truck, 20% chance to leave unassigned
            if random.random() < 0.8:
                kromosom.barang_ke_truk[i] = random.randint(0, self.jumlah_truk - 1)
            else:
                kromosom.barang_ke_truk[i] = -1  # Unassigned

        return kromosom

    def _tournament_selection(self, population, tournament_size=3):
        """Tournament selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda k: k.fitness)

    def _crossover(self, parent1: Kromosom, parent2: Kromosom) -> Kromosom:
        """Single point crossover"""
        child = Kromosom(self.jumlah_barang, self.jumlah_truk)
        crossover_point = random.randint(1, self.jumlah_barang - 1)

        child.barang_ke_truk[:crossover_point] = parent1.barang_ke_truk[:crossover_point]
        child.barang_ke_truk[crossover_point:] = parent2.barang_ke_truk[crossover_point:]

        return child

    def _mutate(self, kromosom: Kromosom, mutation_rate=0.1) -> Kromosom:
        """Mutation operation"""
        for i in range(self.jumlah_barang):
            if random.random() < mutation_rate:
                if random.random() < 0.1:  # 10% chance to unassign
                    kromosom.barang_ke_truk[i] = -1
                else:
                    kromosom.barang_ke_truk[i] = random.randint(0, self.jumlah_truk - 1)

        return kromosom

    def evaluate_kromosom(self, kromosom: Kromosom) -> None:
        """Enhanced evaluation that considers network routing"""
        total_profit = 0.0
        total_cost = 0.0
        penalti = 0.0
        penalti_barang_tidak_dimuat = 0.0

        kromosom.urutan_kota_per_truk = [[] for _ in range(self.jumlah_truk)]
        kromosom.routing_details = [None for _ in range(self.jumlah_truk)]

        truk_loads = {i: {'berat': 0, 'dimensi': 0, 'kota': set(), 'barang': []}
                      for i in range(self.jumlah_truk)}

        # Assign items to trucks
        for barang_idx, truk_idx in enumerate(kromosom.barang_ke_truk):
            barang = self.barang_list[barang_idx]

            if truk_idx >= 0:
                truk = self.truk_list[truk_idx]

                if not truk.can_deliver_to(barang.kota_tujuan):
                    penalti += 1e6
                    penalti_barang_tidak_dimuat += barang.base_total_profit * 1.5
                    continue

                truk_loads[truk_idx]['berat'] += barang.berat
                truk_loads[truk_idx]['dimensi'] += barang.dimensi
                truk_loads[truk_idx]['kota'].add(barang.kota_tujuan)
                truk_loads[truk_idx]['barang'].append(barang)

                distance = self.barang_distances.get((barang.id, truk.kota_asal, barang.kota_tujuan), 1000)
                profit = barang.calculate_profit(distance)
                total_profit += profit
            else:
                # Calculate penalty for unloaded items
                valid_trucks = [t for t in self.truk_list if t.can_deliver_to(barang.kota_tujuan)]
                if valid_trucks:
                    avg_distance = sum(self.barang_distances.get((barang.id, t.kota_asal, barang.kota_tujuan), 1000)
                                       for t in valid_trucks) / len(valid_trucks)
                    lost_profit = barang.calculate_profit(avg_distance)
                    penalti_barang_tidak_dimuat += lost_profit * 1.2

        # Calculate routes and costs for each truck
        for truk_idx, truk in enumerate(self.truk_list):
            load = truk_loads[truk_idx]

            # Check capacity constraints
            if load['berat'] > truk.kapasitas_berat:
                excess_weight = load['berat'] - truk.kapasitas_berat
                penalti += excess_weight * 1e4

            if load['dimensi'] > truk.kapasitas_dimensi:
                excess_volume = load['dimensi'] - truk.kapasitas_dimensi
                penalti += excess_volume * 1e4

            # Calculate route if truck has destinations
            if load['kota']:
                kota_list = list(load['kota'])

                if self.use_network_routing and self.aco_solver:
                    # Use enhanced ACO with routing details
                    best_route, total_distance, routing_info = self.aco_solver.solve_tsp_with_routing(
                        kota_list, truk.kota_asal
                    )
                    kromosom.urutan_kota_per_truk[truk_idx] = best_route
                    kromosom.routing_details[truk_idx] = routing_info
                else:
                    # Simple calculation without routing
                    total_distance = sum(self.barang_distances.get((b.id, truk.kota_asal, b.kota_tujuan), 1000)
                                         for b in load['barang'])
                    kromosom.urutan_kota_per_truk[truk_idx] = kota_list
                    kromosom.routing_details[truk_idx] = {'total_distance': total_distance}

                # Calculate fuel cost
                biaya_bbm = total_distance * truk.konsumsi_bbm * self.harga_bbm / 100
                total_cost += biaya_bbm

        # Set final values
        kromosom.total_profit = total_profit
        kromosom.total_cost = total_cost
        kromosom.penalti = penalti
        kromosom.penalti_barang_tidak_dimuat = penalti_barang_tidak_dimuat
        kromosom.is_valid = penalti == 0
        kromosom.fitness = total_profit - total_cost - penalti - penalti_barang_tidak_dimuat


def create_realistic_indonesia_network():
    """Create a realistic network of Indonesian cities"""
    connections = {
        # Jakarta as major hub
        ("Jakarta", "Bandung"): 150,
        ("Jakarta", "Semarang"): 450,
        ("Jakarta", "Yogyakarta"): 560,
        ("Jakarta", "Surabaya"): 800,
        ("Jakarta", "Medan"): 1400,
        ("Jakarta", "Palembang"): 550,
        ("Jakarta", "Denpasar"): 1150,

        # Surabaya as eastern hub
        ("Surabaya", "Malang"): 90,
        ("Surabaya", "Kediri"): 130,
        ("Surabaya", "Jember"): 200,
        ("Surabaya", "Denpasar"): 350,
        ("Surabaya", "Makassar"): 650,

        # Medan as northern hub
        ("Medan", "Padang"): 450,
        ("Medan", "Pekanbaru"): 350,
        ("Medan", "Banda Aceh"): 400,

        # Makassar as eastern hub
        ("Makassar", "Manado"): 750,
        ("Makassar", "Ambon"): 650,
        ("Makassar", "Jayapura"): 1200,

        # Regional connections
        ("Semarang", "Yogyakarta"): 120,
        ("Semarang", "Surabaya"): 350,
        ("Bandung", "Yogyakarta"): 400,
        ("Palembang", "Padang"): 400,
        ("Pekanbaru", "Padang"): 200,
        ("Denpasar", "Mataram"): 150,
    }

    # Make bidirectional
    full_network = {}
    for (city1, city2), distance in connections.items():
        full_network[(city1, city2)] = distance
        full_network[(city2, city1)] = distance

    return full_network


# Initialize default network
network_matrix = create_realistic_indonesia_network()


@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html',
                           barang_count=len(barang_list),
                           truk_count=len(truk_list),
                           has_results=bool(optimization_results))


@app.route('/barang')
def barang_management():
    """Barang management page"""
    return render_template('barang.html', barang_list=barang_list)


@app.route('/api/barang', methods=['GET'])
def get_barang():
    """Get all barang data"""
    return jsonify([barang.to_dict() for barang in barang_list])


@app.route('/api/barang', methods=['POST'])
def add_barang():
    """Add new barang"""
    try:
        data = request.json
        barang = Barang(
            id=data['id'],
            berat=float(data['berat']),
            dimensi=float(data['dimensi']),
            kota_tujuan=data['kota_tujuan'],
            base_price=float(data['base_price']),
            urgency=int(data['urgency'])
        )

        # Check if ID already exists
        if any(b.id == barang.id for b in barang_list):
            return jsonify({'error': 'Barang ID already exists'}), 400

        barang_list.append(barang)
        return jsonify({'message': 'Barang added successfully', 'barang': barang.to_dict()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/barang/<barang_id>', methods=['PUT'])
def update_barang(barang_id):
    """Update existing barang"""
    try:
        data = request.json

        # Find barang to update
        barang_to_update = None
        for barang in barang_list:
            if barang.id == barang_id:
                barang_to_update = barang
                break

        if not barang_to_update:
            return jsonify({'error': 'Barang not found'}), 404

        # Update barang properties
        barang_to_update.berat = float(data['berat'])
        barang_to_update.dimensi = float(data['dimensi'])
        barang_to_update.kota_tujuan = data['kota_tujuan']
        barang_to_update.base_price = float(data['base_price'])
        barang_to_update.urgency = int(data['urgency'])
        barang_to_update.base_total_profit = barang_to_update.base_price * (1 + barang_to_update.urgency * 0.2)

        return jsonify({'message': 'Barang updated successfully', 'barang': barang_to_update.to_dict()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/barang/<barang_id>', methods=['DELETE'])
def delete_barang(barang_id):
    """Delete barang"""
    global barang_list

    barang_list = [b for b in barang_list if b.id != barang_id]
    return jsonify({'message': 'Barang deleted successfully'})

@app.route('/truk')
def truk_management():
    """Truk management page"""
    return render_template('truk.html', truk_list=truk_list)

@app.route('/api/truk', methods=['GET'])
def get_truk():
    """Get all truk data"""
    return jsonify([truk.to_dict() for truk in truk_list])

@app.route('/api/truk', methods=['POST'])
def add_truk():
    """Add new truk"""
    try:
        data = request.json
        truk = Truk(
            id=int(data['id']),
            kapasitas_berat=float(data['kapasitas_berat']),
            kapasitas_dimensi=float(data['kapasitas_dimensi']),
            konsumsi_bbm=float(data['konsumsi_bbm']),
            kota_asal=data['kota_asal']
        )

        # Check if ID already exists
        if any(t.id == truk.id for t in truk_list):
            return jsonify({'error': 'Truk ID already exists'}), 400

        truk_list.append(truk)
        return jsonify({'message': 'Truk added successfully', 'truk': truk.to_dict()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/truk/<int:truk_id>', methods=['PUT'])
def update_truk(truk_id):
    """Update existing truk"""
    try:
        data = request.json

        # Find truk to update
        truk_to_update = None
        for truk in truk_list:
            if truk.id == truk_id:
                truk_to_update = truk
                break

        if not truk_to_update:
            return jsonify({'error': 'Truk not found'}), 404

        # Update truk properties
        truk_to_update.kapasitas_berat = float(data['kapasitas_berat'])
        truk_to_update.kapasitas_dimensi = float(data['kapasitas_dimensi'])
        truk_to_update.konsumsi_bbm = float(data['konsumsi_bbm'])
        truk_to_update.kota_asal = data['kota_asal']

        return jsonify({'message': 'Truk updated successfully', 'truk': truk_to_update.to_dict()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/truk/<int:truk_id>', methods=['DELETE'])
def delete_truk(truk_id):
    """Delete truk"""
    global truk_list

    truk_list = [t for t in truk_list if t.id != truk_id]
    return jsonify({'message': 'Truk deleted successfully'})

@app.route('/network')
def network_management():
    """Network management page"""
    return render_template('network.html', network_matrix=network_matrix)

@app.route('/api/network', methods=['GET'])
def get_network():
    """Get network matrix"""
    # Convert tuple keys to string format for JSON
    network_data = {}
    for (city1, city2), distance in network_matrix.items():
        key = f"{city1}-{city2}"
        network_data[key] = distance

    return jsonify(network_data)

@app.route('/api/network', methods=['POST'])
def add_network_connection():
    """Add network connection"""
    try:
        data = request.json
        city1 = data['city1']
        city2 = data['city2']
        distance = float(data['distance'])

        # Add bidirectional connection
        network_matrix[(city1, city2)] = distance
        network_matrix[(city2, city1)] = distance

        return jsonify({'message': 'Connection added successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/network/<city1>/<city2>', methods=['DELETE'])
def delete_network_connection(city1, city2):
    """Delete network connection"""
    global network_matrix

    # Remove both directions
    network_matrix.pop((city1, city2), None)
    network_matrix.pop((city2, city1), None)

    return jsonify({'message': 'Connection deleted successfully'})

def optimization_worker(population_size, generations):
    """Background optimization worker"""
    global optimization_status, optimization_results

    try:
        optimization_status["running"] = True
        optimization_status["progress"] = 0
        optimization_status["current_generation"] = 0

        def progress_callback(current_gen, total_gen):
            optimization_status["current_generation"] = current_gen
            optimization_status["progress"] = int((current_gen / total_gen) * 100)

        # Create optimizer
        optimizer = OptimizedGAACOOptimizer(
            barang_list=barang_list,
            truk_list=truk_list,
            jarak_matrix=network_matrix,
            use_network_routing=True
        )

        # Run optimization
        best_solution = optimizer.optimize(
            population_size=population_size,
            generations=generations,
            callback=progress_callback
        )

        # Convert solution to dictionary format for JSON serialization
        optimization_results = {
            'fitness': best_solution.fitness,
            'total_profit': best_solution.total_profit,
            'total_cost': best_solution.total_cost,
            'penalti': best_solution.penalti,
            'penalti_barang_tidak_dimuat': best_solution.penalti_barang_tidak_dimuat,
            'is_valid': best_solution.is_valid,
            'barang_ke_truk': best_solution.barang_ke_truk,
            'urutan_kota_per_truk': best_solution.urutan_kota_per_truk,
            'routing_details': best_solution.routing_details,
            'truck_assignments': []
        }

        # Create detailed truck assignments
        for truk_idx, truk in enumerate(truk_list):
            assigned_barang = []
            total_berat = 0
            total_dimensi = 0

            for barang_idx, assigned_truk_idx in enumerate(best_solution.barang_ke_truk):
                if assigned_truk_idx == truk_idx:
                    barang = barang_list[barang_idx]
                    assigned_barang.append({
                        'id': barang.id,
                        'berat': barang.berat,
                        'dimensi': barang.dimensi,
                        'kota_tujuan': barang.kota_tujuan,
                        'base_price': barang.base_price,
                        'urgency': barang.urgency
                    })
                    total_berat += barang.berat
                    total_dimensi += barang.dimensi

            route_info = best_solution.routing_details[truk_idx] if truk_idx < len(
                best_solution.routing_details) else None

            optimization_results['truck_assignments'].append({
                'truk_id': truk.id,
                'truk_data': truk.to_dict(),
                'assigned_barang': assigned_barang,
                'total_berat': total_berat,
                'total_dimensi': total_dimensi,
                'route': best_solution.urutan_kota_per_truk[truk_idx] if truk_idx < len(
                    best_solution.urutan_kota_per_truk) else [],
                'routing_details': route_info,
                'capacity_utilization': {
                    'berat': (total_berat / truk.kapasitas_berat) * 100 if truk.kapasitas_berat > 0 else 0,
                    'dimensi': (total_dimensi / truk.kapasitas_dimensi) * 100 if truk.kapasitas_dimensi > 0 else 0
                }
            })

        optimization_status["progress"] = 100
        optimization_status["running"] = False

    except Exception as e:
        optimization_status["running"] = False
        optimization_status["error"] = str(e)
        print(f"Optimization error: {e}")

@app.route('/optimize')
def optimize_page():
    """Optimization page"""
    return render_template('optimize.html')

@app.route('/api/optimize', methods=['POST'])
def start_optimization():
    """Start optimization process"""
    if optimization_status["running"]:
        return jsonify({'error': 'Optimization already running'}), 400

    if not barang_list or not truk_list:
        return jsonify({'error': 'Please add barang and truk data first'}), 400

    try:
        data = request.json or {}
        population_size = int(data.get('population_size', 50))
        generations = int(data.get('generations', 100))

        # Start optimization in background thread
        thread = threading.Thread(
            target=optimization_worker,
            args=(population_size, generations)
        )
        thread.daemon = True
        thread.start()

        return jsonify({'message': 'Optimization started'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/optimize/status', methods=['GET'])
def get_optimization_status():
    """Get optimization status"""
    return jsonify(optimization_status)

@app.route('/api/optimize/stop', methods=['POST'])
def stop_optimization():
    """Stop optimization process"""
    optimization_status["running"] = False
    return jsonify({'message': 'Optimization stopped'})

@app.route('/results')
def results_page():
    """Results page"""
    if not optimization_results:
        flash('No optimization results available. Please run optimization first.', 'warning')
        return redirect(url_for('optimize_page'))

    return render_template('results.html', results=optimization_results)

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get optimization results"""
    if not optimization_results:
        return jsonify({'error': 'No results available'}), 404

    return jsonify(optimization_results)

@app.route('/api/results/export', methods=['GET'])
def export_results():
    """Export results to JSON"""
    if not optimization_results:
        return jsonify({'error': 'No results available'}), 404

    # Create export data
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'optimization_results': optimization_results,
        'barang_data': [b.to_dict() for b in barang_list],
        'truk_data': [t.to_dict() for t in truk_list],
        'network_data': {f"{k[0]}-{k[1]}": v for k, v in network_matrix.items()}
    }

    return jsonify(export_data)

@app.route('/api/data/clear', methods=['POST'])
def clear_all_data():
    """Clear all data"""
    global barang_list, truk_list, optimization_results, optimization_status

    barang_list.clear()
    truk_list.clear()
    optimization_results.clear()
    optimization_status = {"running": False, "progress": 0, "current_generation": 0}

    return jsonify({'message': 'All data cleared successfully'})

@app.route('/api/data/sample', methods=['POST'])
def load_sample_data():
    """Load sample data for testing"""
    global barang_list, truk_list

    # Clear existing data
    barang_list.clear()
    truk_list.clear()

    # Sample barang data
    sample_barang = [
        Barang("B001", 100, 50, "Surabaya", 500000, 2),
        Barang("B002", 150, 75, "Medan", 750000, 3),
        Barang("B003", 80, 40, "Denpasar", 400000, 1),
        Barang("B004", 200, 100, "Makassar", 800000, 2),
        Barang("B005", 120, 60, "Bandung", 600000, 3),
        Barang("B006", 90, 45, "Semarang", 450000, 1),
        Barang("B007", 160, 80, "Yogyakarta", 650000, 2),
        Barang("B008", 110, 55, "Malang", 550000, 3),
    ]
    barang_list.extend(sample_barang)

    # Sample truk data
    sample_truk = [
        Truk(1, 1000, 500, 12, "Jakarta"),
        Truk(2, 1200, 600, 15, "Jakarta"),
        Truk(3, 800, 400, 10, "Surabaya"),
        Truk(4, 1500, 750, 18, "Jakarta"),
    ]
    truk_list.extend(sample_truk)

    return jsonify({
        'message': 'Sample data loaded successfully',
        'barang_count': len(barang_list),
        'truk_count': len(truk_list)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
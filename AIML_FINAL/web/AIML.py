from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import threading
from datetime import datetime
import random
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict
import os
import requests # Library untuk memanggil API

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

def get_coordinates_for_cities(cities: List[str], api_key: str = None) -> Tuple[Dict[str, Dict], str | None]:
    print("[DEBUG] Masuk ke fungsi get_coordinates_for_cities [NOMINATIM VERSION]")
    coordinates = {}
    for city in cities:
        city_norm = city.strip().title()
        if not city_norm: continue
        print(f"[DEBUG] Memproses kota: {city_norm}")
        try:
            url = f"https://nominatim.openstreetmap.org/search"
            params = {'q': f"{city_norm}, Indonesia", 'format': 'json', 'limit': 1}
            headers = {'User-Agent': 'FlaskAppGeocoder/1.0 (contact@example.com)'}
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data:
                coordinates[city_norm] = {
                    'latitude': float(data[0]['lat']),
                    'longitude': float(data[0]['lon'])
                }
                print(f"[SUCCESS] Koordinat {city_norm}: {coordinates[city_norm]}")
            else:
                return {}, f"Gagal menemukan koordinat untuk kota '{city_norm}'. Pastikan nama kota valid."
        except Exception as e:
            return {}, f"Gagal mendapatkan koordinat untuk '{city_norm}': {e}"
    return coordinates, None

def build_distance_matrix_from_coords(coordinates: Dict[str, Dict], api_key: str) -> Tuple[Dict[Tuple[str, str], float], str | None]:
    print("[DEBUG] Membangun Matriks Jarak All-Pairs...")
    jarak_matrix = {}
    distance_url = 'https://distance-calculator.p.rapidapi.com/v1/one_to_one'
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "distance-calculator.p.rapidapi.com",
        "Content-Type": "application/json"
    }
    cities = list(coordinates.keys())
    for city in cities:
        jarak_matrix[(city, city)] = 0.0
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            city1, city2 = cities[i], cities[j]
            try:
                lat1, lon1 = coordinates[city1]['latitude'], coordinates[city1]['longitude']
                lat2, lon2 = coordinates[city2]['latitude'], coordinates[city2]['longitude']
                querystring = {"start_point": f"({lat1},{lon1})", "end_point": f"({lat2},{lon2})", "unit": "kilometers", "decimal_places": "2"}
                response = requests.get(distance_url, headers=headers, params=querystring, timeout=15)
                response.raise_for_status()
                data = response.json()
                if 'distance' in data:
                    distance_km = float(data['distance'])
                    jarak_matrix[(city1, city2)] = distance_km
                    jarak_matrix[(city2, city1)] = distance_km
                    print(f"[SUCCESS] Jarak {city1} <-> {city2}: {distance_km} km")
                else:
                    return {}, f"Gagal menghitung jarak antara {city1} dan {city2}. Detail: {data.get('error', 'Unknown API error')}"
            except requests.exceptions.RequestException as e:
                return {}, f"Gagal terhubung ke Distance API saat menghitung {city1}-{city2}: {e}"
            except Exception as e:
                return {}, f"Error saat memproses jarak untuk {city1}-{city2}: {e}"
    print("[SUCCESS] Matriks Jarak lengkap berhasil dibuat.")
    return jarak_matrix, None

@dataclass
class Barang:
    id: str
    berat: float
    dimensi: float
    kota_tujuan: str
    profit_per_kg: float

    @property
    def total_profit(self) -> float:
        return self.berat * 5000

    @property
    def value_density(self) -> float:
        denominator = self.berat + self.dimensi + 1e-6 
        return self.total_profit / denominator

@dataclass
class Truk:
    id: int
    kapasitas_berat: float
    kapasitas_dimensi: float
    konsumsi_bbm: float
    kota_asal: str = "Jakarta"

# victor
class Kromosom:
    def __init__(self, jumlah_barang: int, jumlah_truk: int):
        self.barang_ke_truk = [-1] * jumlah_barang
        self.urutan_kota_per_truk = [[] for _ in range(jumlah_truk)]
        self.loading_order_per_truk = [[] for _ in range(jumlah_truk)]
        self.fitness = -float('inf')
        self.total_profit = 0.0
        self.total_cost = 0.0
        self.penalti = 0.0
        self.penalti_barang_tidak_dimuat = 0.0
        self.is_valid = False

# jean
class ImprovedACOSolver:
    def __init__(self, jarak_matrix: Dict[Tuple[str, str], float],
                 alpha=1.0, beta=5.0, rho=0.5, n_ants=20, n_iterations=50):
        self.jarak_matrix, self.alpha, self.beta, self.rho = jarak_matrix, alpha, beta, rho
        self.n_ants, self.n_iterations = n_ants, n_iterations
        self.pheromone = defaultdict(lambda: 1.0)
        self.best_solution_cache = {}

    def solve_tsp(self, kota_list: List[str], kota_asal: str = "Jakarta") -> Tuple[List[str], float]:
        if not kota_list: return [], 0.0
        
        cache_key = (kota_asal, tuple(sorted(kota_list)))
        if cache_key in self.best_solution_cache:
            return self.best_solution_cache[cache_key]

        if len(kota_list) == 1:
            kota_tujuan = kota_list[0]
            jarak = self.jarak_matrix.get((kota_asal, kota_tujuan), 99999) + self.jarak_matrix.get((kota_tujuan, kota_asal), 99999)
            self.best_solution_cache[cache_key] = (kota_list, jarak)
            return kota_list, jarak
        
        # Inisialisasi
        all_cities_for_tsp = [kota_asal] + kota_list
        best_route_overall, best_dist_overall = self._nearest_neighbor(kota_list, kota_asal)
        self._init_pheromone(nn_dist=best_dist_overall, cities=all_cities_for_tsp)

        # Iterasi ACO
        for _ in range(self.n_iterations):
            all_routes_in_iter = []
            for _ in range(self.n_ants):
                route, dist = self._construct_route(kota_list, kota_asal)
                if dist < best_dist_overall:
                    best_dist_overall, best_route_overall = dist, route
                all_routes_in_iter.append({'route': route, 'dist': dist})
            self._update_pheromone(all_routes_in_iter, best_route_overall, best_dist_overall, kota_asal)
        
        final_route, final_dist = self._2_opt(best_route_overall, best_dist_overall, kota_asal)

        self.best_solution_cache[cache_key] = (final_route, final_dist)
        return final_route, final_dist
    
    def _init_pheromone(self, nn_dist, cities):
        p_val = 1.0 / nn_dist if nn_dist > 0 else 1.0
        for i in range(len(cities)):
            for j in range(i + 1, len(cities)):
                self.pheromone[(cities[i], cities[j])] = p_val
                self.pheromone[(cities[j], cities[i])] = p_val

    def _construct_route(self, unvisited_cities: List[str], kota_asal: str) -> Tuple[List[str], float]:
        route, unvisited, current_city, total_distance = [], unvisited_cities.copy(), kota_asal, 0
        while unvisited:
            next_city = self._select_next_city(current_city, unvisited)
            route.append(next_city)
            total_distance += self.jarak_matrix.get((current_city, next_city), 99999)
            unvisited.remove(next_city)
            current_city = next_city
        total_distance += self.jarak_matrix.get((current_city, kota_asal), 99999)
        return route, total_distance

    def _select_next_city(self, current_city: str, unvisited: List[str]) -> str:
        probabilities = []
        for city in unvisited:
            jarak = self.jarak_matrix.get((current_city, city), 99999)
            heuristic = (1.0 / jarak) if jarak > 0 else 1e-10
            prob = (self.pheromone[(current_city, city)] ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(prob)
        total_prob = sum(probabilities)
        if total_prob == 0: return random.choice(unvisited)
        norm_probs = [p / total_prob for p in probabilities]
        return np.random.choice(unvisited, p=norm_probs)

    def _update_pheromone(self, routes_info, best_route, best_dist, kota_asal):
        # Evaporasi
        for key in self.pheromone: self.pheromone[key] *= (1 - self.rho)
        # Deposisi oleh semua semut
        for info in routes_info:
            if info['dist'] > 0:
                p_deposit = 1.0 / info['dist']
                full_route = [kota_asal] + info['route'] + [kota_asal]
                for i in range(len(full_route) - 1):
                    self.pheromone[(full_route[i], full_route[i+1])] += p_deposit
                    self.pheromone[(full_route[i+1], full_route[i])] += p_deposit
        # Deposisi bonus oleh semut elit
        if best_dist > 0:
            elite_deposit = (1.0 / best_dist) * self.n_ants # Bonus feromon
            full_best_route = [kota_asal] + best_route + [kota_asal]
            for i in range(len(full_best_route) - 1):
                self.pheromone[(full_best_route[i], full_best_route[i+1])] += elite_deposit
                self.pheromone[(full_best_route[i+1], full_best_route[i])] += elite_deposit
    
    def _nearest_neighbor(self, kota_list: List[str], kota_asal: str) -> Tuple[List[str], float]:
        route, unvisited, current_city, total_distance = [], kota_list.copy(), kota_asal, 0
        while unvisited:
            nearest_city = min(unvisited, key=lambda city: self.jarak_matrix.get((current_city, city), 99999))
            route.append(nearest_city)
            total_distance += self.jarak_matrix.get((current_city, nearest_city), 99999)
            unvisited.remove(nearest_city)
            current_city = nearest_city
        total_distance += self.jarak_matrix.get((current_city, kota_asal), 99999)
        return route, total_distance
    
    def _2_opt(self, route: List[str], distance: float, kota_asal: str) -> Tuple[List[str], float]:
        """Penyempurnaan rute menggunakan 2-Opt local search."""
        best_route, best_distance = route, distance
        full_route = [kota_asal] + route + [kota_asal]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(full_route) - 2):
                for j in range(i + 1, len(full_route)):
                    if j == i + 1: continue
                    current_dist = self.jarak_matrix.get((full_route[i-1], full_route[i]), 99999) + \
                                   self.jarak_matrix.get((full_route[j-1], full_route[j]), 99999)
                    new_dist = self.jarak_matrix.get((full_route[i-1], full_route[j-1]), 99999) + \
                               self.jarak_matrix.get((full_route[i], full_route[j]), 99999)
                    if new_dist < current_dist:
                        # Lakukan pertukaran (swap)
                        new_full_route = full_route[:i] + full_route[i:j][::-1] + full_route[j:]
                        full_route = new_full_route
                        best_distance = best_distance - current_dist + new_dist
                        improved = True
        return full_route[1:-1], best_distance

# victor
class OptimizedGAACOOptimizer:
    def __init__(self, barang_list: List[Barang], truk_list: List[Truk],
                 jarak_matrix: Dict[Tuple[str, str], float], harga_bbm: float = 15000):
        self.barang_list, self.truk_list, self.jarak_matrix, self.harga_bbm = barang_list, truk_list, jarak_matrix, harga_bbm
        self.jumlah_barang, self.jumlah_truk = len(barang_list), len(truk_list)
        # NOTE: Menggunakan solver yang sudah disempurnakan
        self.aco_solver = ImprovedACOSolver(jarak_matrix)
        self.sorted_barang_by_value = sorted(enumerate(barang_list), key=lambda x: x[1].value_density, reverse=True)
        self.route_cache = {}

    def create_smart_kromosom(self) -> Kromosom:
        kromosom = Kromosom(self.jumlah_barang, self.jumlah_truk)
        truk_sisa_berat = [t.kapasitas_berat for t in self.truk_list]
        truk_sisa_dimensi = [t.kapasitas_dimensi for t in self.truk_list]
        for barang_idx, barang in self.sorted_barang_by_value:
            possible_trucks = []
            for truk_idx in range(self.jumlah_truk):
                if barang.berat <= truk_sisa_berat[truk_idx] and barang.dimensi <= truk_sisa_dimensi[truk_idx]:
                    sisa = (truk_sisa_berat[truk_idx] - barang.berat) + (truk_sisa_dimensi[truk_idx] - barang.dimensi)
                    possible_trucks.append((sisa, truk_idx))
            if possible_trucks:
                best_truk_idx = min(possible_trucks, key=lambda x: x[0])[1]
                kromosom.barang_ke_truk[barang_idx] = best_truk_idx
                truk_sisa_berat[best_truk_idx] -= barang.berat
                truk_sisa_dimensi[best_truk_idx] -= barang.dimensi
        return kromosom

    def create_random_kromosom(self) -> Kromosom:
        kromosom = Kromosom(self.jumlah_barang, self.jumlah_truk)
        for i in range(self.jumlah_barang):
            kromosom.barang_ke_truk[i] = random.randint(-1, self.jumlah_truk - 1)
        return kromosom

    def evaluate_kromosom(self, kromosom: Kromosom):
        """Mengevaluasi kromosom dengan fokus pada profit bersih yang maksimal."""
        # 1. Inisialisasi semua komponen
        total_profit_kotor = 0.0
        total_biaya_operasional = 0.0
        total_penalti_kapasitas = 0.0
        total_penalti_tidak_dimuat = 0.0

        kromosom.urutan_kota_per_truk = [[] for _ in range(self.jumlah_truk)]
        kromosom.loading_order_per_truk = [[] for _ in range(self.jumlah_truk)]
        
        # Mengelompokkan barang berdasarkan penugasan truk
        truk_loads = {i: {'berat': 0, 'dimensi': 0, 'kota': set(), 'barang_indices': []} for i in range(self.jumlah_truk)}
        
        for barang_idx, truk_idx in enumerate(kromosom.barang_ke_truk):
            barang = self.barang_list[barang_idx]
            if truk_idx != -1:
                # Jika barang dimuat, tambahkan ke profit kotor dan beban truk
                total_profit_kotor += barang.total_profit
                truk_loads[truk_idx]['berat'] += barang.berat
                truk_loads[truk_idx]['dimensi'] += barang.dimensi
                truk_loads[truk_idx]['kota'].add(barang.kota_tujuan)
                truk_loads[truk_idx]['barang_indices'].append(barang_idx)
            else:
                # Jika barang tidak dimuat, berikan penalti
                total_penalti_tidak_dimuat += barang.total_profit * 1.2
        
        # 2. Hitung biaya dan penalti untuk setiap truk
        for truk_idx, truk in enumerate(self.truk_list):
            load = truk_loads[truk_idx]

            # Penalti jika kapasitas terlampaui (penalti "hukuman mati")
            if load['berat'] > truk.kapasitas_berat:
                total_penalti_kapasitas += 1e9
            if load['dimensi'] > truk.kapasitas_dimensi:
                total_penalti_kapasitas += 1e9

            # Jika truk ini membawa barang, hitung biaya rutenya
            if load['kota']:
                kota_list = list(load['kota'])
                cache_key = (truk.kota_asal, tuple(sorted(kota_list)))
                
                if cache_key in self.route_cache:
                    best_route, total_distance = self.route_cache[cache_key]
                else:
                    best_route, total_distance = self.aco_solver.solve_tsp(kota_list, truk.kota_asal)
                    self.route_cache[cache_key] = (best_route, total_distance)
                
                kromosom.urutan_kota_per_truk[truk_idx] = best_route
                if best_route:
                    city_order_map = {city: i for i, city in enumerate(best_route)}
                    sorted_indices = sorted(
                        load['barang_indices'],
                        key=lambda idx: city_order_map.get(self.barang_list[idx].kota_tujuan, -1),
                        reverse=True
                    )
                    kromosom.loading_order_per_truk[truk_idx] = [self.barang_list[i].id for i in sorted_indices]
                
                biaya_bbm = total_distance * truk.konsumsi_bbm * self.harga_bbm / 10
                total_biaya_operasional += biaya_bbm

        # 3. Hitung skor fitness akhir
        kromosom.total_profit = total_profit_kotor
        kromosom.total_cost = total_biaya_operasional
        kromosom.penalti = total_penalti_kapasitas
        kromosom.penalti_barang_tidak_dimuat = total_penalti_tidak_dimuat
        kromosom.is_valid = (total_penalti_kapasitas == 0)
        
        # Rumus utama untuk optimasi profit bersih
        kromosom.fitness = total_profit_kotor - total_biaya_operasional - total_penalti_kapasitas - total_penalti_tidak_dimuat
    
    def optimize(self, population_size: int = 50, generations: int = 50, mutation_rate: float = 0.2, elite_size: int = 10) -> Kromosom:
        populasi = [self.create_smart_kromosom() for _ in range(int(population_size * 0.8))] + \
                   [self.create_random_kromosom() for _ in range(int(population_size * 0.2))]
        for kromosom in populasi: self.evaluate_kromosom(kromosom)
        best_fitness_ever, stagnation_counter = -float('inf'), 0
        for gen in range(generations):
            populasi.sort(key=lambda x: x.fitness, reverse=True)
            print(f"Generasi {gen+1} | Fitness Terbaik: {populasi[0].fitness:.2f}")
            if populasi[0].fitness > best_fitness_ever:
                best_fitness_ever = populasi[0].fitness
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            if stagnation_counter > 10:
                print("Stagnasi terdeteksi, injeksi individu baru...")
                for i in range(int(population_size * 0.2)):
                    populasi[population_size - 1 - i] = self.create_smart_kromosom()
                    self.evaluate_kromosom(populasi[-(i+1)])
                stagnation_counter = 0
            new_population = populasi[:elite_size]
            while len(new_population) < population_size:
                parent1 = self.tournament_selection(populasi)
                parent2 = self.tournament_selection(populasi)
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1, mutation_rate)
                self.mutate(child2, mutation_rate)
                self.evaluate_kromosom(child1)
                self.evaluate_kromosom(child2)
                new_population.extend([child1, child2])
            populasi = new_population[:population_size]
        populasi.sort(key=lambda x: x.fitness, reverse=True)
        return populasi[0]

    def tournament_selection(self, populasi: List[Kromosom], k: int = 3) -> Kromosom:
        return max(random.sample(populasi, k), key=lambda x: x.fitness)

    def crossover(self, parent1: Kromosom, parent2: Kromosom) -> Tuple[Kromosom, Kromosom]:
        child1, child2 = Kromosom(self.jumlah_barang, self.jumlah_truk), Kromosom(self.jumlah_barang, self.jumlah_truk)
        point = random.randint(1, self.jumlah_barang - 1)
        child1.barang_ke_truk = parent1.barang_ke_truk[:point] + parent2.barang_ke_truk[point:]
        child2.barang_ke_truk = parent2.barang_ke_truk[:point] + parent1.barang_ke_truk[point:]
        return child1, child2

    def mutate(self, kromosom: Kromosom, mutation_rate: float):
        for i in range(self.jumlah_barang):
            if random.random() < mutation_rate:
                kromosom.barang_ke_truk[i] = random.randint(-1, self.jumlah_truk - 1)

current_data = {'barang_list': [], 'truk_list': [], 'jarak_matrix': {}, 'harga_bbm': 15000}
optimization_results = {}
optimization_thread = None
optimization_status = {'running': False, 'completed': False, 'error': None}


# route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input_data')
def input_data():
    return render_template('input_data.html')

@app.route('/api/data_summary')
def get_data_summary():
    """Endpoint untuk memberikan ringkasan data yang sudah di-load."""
    if not current_data['barang_list'] or not current_data['truk_list']:
        return jsonify({'status': 'success', 'data': {'has_data': False}})

    barang_list = current_data['barang_list']
    truk_list = current_data['truk_list']
    
    potential_profit = sum(b.total_profit for b in barang_list)
    cities = list(set(b.kota_tujuan for b in barang_list))

    summary = {
        'has_data': True,
        'barang_count': len(barang_list),
        'truk_count': len(truk_list),
        'potential_profit': potential_profit,
        'cities': cities
    }
    return jsonify({'status': 'success', 'data': summary})

@app.route('/api/save_data', methods=['POST'])
def save_data():
    try:
        RAPIDAPI_KEY = "9b89da684cmsh620913a3417a2ddp17984bjsn014532dfb106"
        if "GANTI_DENGAN" in RAPIDAPI_KEY:
            return jsonify({'status': 'error', 'message': 'RapidAPI Key belum diatur di dalam kode!'})
        data = request.json
        barang_list = [Barang(id=item['id'], berat=float(item['berat']), dimensi=float(item['dimensi']), kota_tujuan=item['kota_tujuan'].strip().title(), profit_per_kg=float(item['profit_per_kg'])) for item in data['barang']]
        truk_list = [Truk(id=int(item['id']), kapasitas_berat=float(item['kapasitas_berat']), kapasitas_dimensi=float(item['kapasitas_dimensi']), konsumsi_bbm=float(item['konsumsi_bbm']), kota_asal=item.get('kota_asal', 'Jakarta').strip().title()) for item in data['truk']]
        unique_cities_set = set(b.kota_tujuan for b in barang_list) | set(t.kota_asal for t in truk_list)
        unique_cities_set.add("Jakarta")
        unique_cities = list(unique_cities_set)
        coordinates, error_msg = get_coordinates_for_cities(unique_cities)
        if error_msg:
            return jsonify({'status': 'error', 'message': error_msg})
        jarak_matrix, error_msg = build_distance_matrix_from_coords(coordinates, RAPIDAPI_KEY)
        if error_msg:
            return jsonify({'status': 'error', 'message': error_msg})
        current_data.update({
            'barang_list': barang_list, 'truk_list': truk_list,
            'jarak_matrix': jarak_matrix, 'harga_bbm': float(data.get('harga_bbm', 15000))
        })
        return jsonify({'status': 'success', 'message': 'Data berhasil disimpan dan matriks jarak lengkap telah dihitung!'})
    except Exception as e:
        print(f"Error di save_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Terjadi kesalahan internal: {e}'})

@app.route('/optimize')
def optimize_page():
    if not current_data['barang_list'] or not current_data['truk_list']:
        flash('Silakan masukkan data terlebih dahulu!', 'error')
        return redirect(url_for('input_data'))
    return render_template('optimize.html')

@app.route('/api/start_optimization', methods=['POST'])
def start_optimization():
    global optimization_thread
    if optimization_status['running']: return jsonify({'status': 'error', 'message': 'Proses optimasi sedang berjalan.'})
    params = request.json or {}
    def run_optimization():
        global optimization_results, optimization_status
        try:
            optimization_status.update({'running': True, 'completed': False, 'error': None})
            optimizer = OptimizedGAACOOptimizer(current_data['barang_list'], current_data['truk_list'], current_data['jarak_matrix'], current_data['harga_bbm'])
            best_solution = optimizer.optimize(
                population_size=int(params.get('population_size', 50)), 
                generations=int(params.get('generations', 50)), 
                mutation_rate=float(params.get('mutation_rate', 0.2))
            )
            results = {
                'fitness': best_solution.fitness, 'total_profit': best_solution.total_profit, 
                'total_cost': best_solution.total_cost, 'net_profit': best_solution.fitness, 
                'is_valid': best_solution.is_valid, 'penalti': best_solution.penalti + best_solution.penalti_barang_tidak_dimuat, 
                'timestamp': datetime.now().isoformat()
            }
            truck_details, total_loaded_items, item_map = [], 0, {item.id: item for item in current_data['barang_list']}
            for truk_idx, truk in enumerate(current_data['truk_list']):
                # Selalu dapatkan daftar ID barang (akan menjadi list kosong jika tidak ada)
                loading_order_ids = best_solution.loading_order_per_truk[truk_idx]
                
                # Konversi dari ID ke objek barang
                items_in_truck = [item_map[item_id] for item_id in loading_order_ids]
                total_loaded_items += len(items_in_truck)
                truck_details.append({
                    'truk_id': truk.id,
                    'total_weight': sum(item.berat for item in items_in_truck),
                    'capacity_weight': truk.kapasitas_berat,
                    'total_volume': sum(item.dimensi for item in items_in_truck),
                    'capacity_volume': truk.kapasitas_dimensi,
                    'items_count': len(items_in_truck),
                    'profit': sum(item.total_profit for item in items_in_truck),
                    'route': [truk.kota_asal] + best_solution.urutan_kota_per_truk[truk_idx] + [truk.kota_asal],
                    'kota_asal': truk.kota_asal,
                    'items': [{
                        'id': item.id,
                        'berat': item.berat,
                        'dimensi': item.dimensi,
                        'kota_tujuan': item.kota_tujuan,
                        'profit_per_kg': item.profit_per_kg,
                        'profit': item.total_profit 
                    } for item in items_in_truck]
                })
            unloaded_barang_objects = [current_data['barang_list'][i] for i, t_idx in enumerate(best_solution.barang_ke_truk) if t_idx == -1]
            unloaded_profit_loss = sum(item.total_profit for item in unloaded_barang_objects)
            unloaded_items_dicts = [{
                'id': item.id,
                'berat': item.berat,
                'dimensi': item.dimensi,
                'kota_tujuan': item.kota_tujuan,
                'profit_per_kg': item.profit_per_kg,
                'profit': item.total_profit 
            } for item in unloaded_barang_objects]

            results.update({
                'truck_details': truck_details, 
                'unloaded_items': unloaded_items_dicts,
                'unloaded_profit_loss': unloaded_profit_loss,
                'total_items': len(current_data['barang_list']), 
                'loaded_items': total_loaded_items
            })
            optimization_results = results
            optimization_status.update({'running': False, 'completed': True})
        except Exception as e:
            optimization_status.update({'running': False, 'error': str(e)})
            print(f"Error during optimization: {e}")
            import traceback
            traceback.print_exc()
    optimization_results = {}
    optimization_thread = threading.Thread(target=run_optimization)
    optimization_thread.start()
    return jsonify({'status': 'success', 'message': 'Optimasi dimulai'})

@app.route('/api/optimization_status')
def get_optimization_status():
    response = optimization_status.copy()
    if optimization_status['completed']: response['results'] = optimization_results
    return jsonify(response)

@app.route('/results')
def results_page():
    if not optimization_status.get('completed'):
        flash('Tidak ada hasil optimasi yang tersedia atau proses belum selesai.', 'error')
        return redirect(url_for('optimize_page'))
    return render_template('results.html', results=optimization_results)

@app.route('/api/sample_data')
def get_sample_data():
    return jsonify({'barang': [{'id': f'B{i:03}', 'berat': random.randint(50, 100), 'dimensi': round(random.uniform(0.2, 1), 2), 'kota_tujuan': random.choice(['Bandung', 'Surabaya', 'Yogyakarta', 'Malang', 'Serang', 'Solo']), 'profit_per_kg': 5000} for i in range(1, 100)], 'truk': [{'id': 1, 'kapasitas_berat': 2000, 'kapasitas_dimensi': 9.0, 'konsumsi_bbm': 0.15, 'kota_asal': 'Jakarta'}, {'id': 2, 'kapasitas_berat': 1500, 'kapasitas_dimensi': 8.0, 'konsumsi_bbm': 0.18, 'kota_asal': 'Jakarta'}, {'id': 3, 'kapasitas_berat': 1000, 'kapasitas_dimensi': 6.0, 'konsumsi_bbm': 0.12, 'kota_asal': 'Jakarta'}, {'id': 4, 'kapasitas_berat': 10000, 'kapasitas_dimensi': 40, 'konsumsi_bbm': 0.2, 'kota_asal': 'Jakarta'}], 'harga_bbm': 15000})

@app.errorhandler(404)
def not_found_error(error): return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error): return render_template('500.html'), 500

if __name__ == '__main__':
    if not os.path.exists('templates'): os.makedirs('templates')
    if not os.path.exists('static'): os.makedirs('static')
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
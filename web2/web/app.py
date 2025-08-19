from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import json
import threading
import time
from datetime import datetime
import random
import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, asdict
import copy
from collections import defaultdict
import os
import requests # Library untuk memanggil API

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# --- FUNGSI GEOCODING YANG DIPERBAIKI ---
def get_coordinates_for_cities(cities: List[str], api_key: str) -> Tuple[Dict[str, Dict], str | None]:
    """
    Mengubah daftar nama kota menjadi koordinat menggunakan google-api31 di RapidAPI.
    """
    coordinates = {}
    # geocoding_url = 'https://google-api31.p.rapidapi.com/map'
    # headers = {
    #     "x-rapidapi-key": "9b89da684cmsh620913a3417a2ddp17984bjsn014532dfb106",
    #     "x-rapidapi-host": "google-api31.p.rapidapi.com",
    #     "Content-Type": "application/json"
    # }
    
    geocoding_url = "https://google-api31.p.rapidapi.com/map"
    headers = {
        "x-rapidapi-key": "5479f59863mshbbe7111d42baf62p150362jsn7b2a2728ce88",
        "x-rapidapi-host": "google-api31.p.rapidapi.com",
        "Content-Type": "application/json"
    }
    
    print(cities)
    for city in cities:
        print(city)
        # PERBAIKAN: Menggunakan variabel 'city' dari perulangan, bukan nilai hardcode
        params = {
        "text": "universitas kristen petra",
        "place": "Indonesia, " + city,
        "street": "",
        "city": "",
        "country": "Indonesia",
        "state": "",
        "postalcode": "",
        "latitude": "",
        "longitude": "",
        "radius": ""
        }
        
        try:
            # PERBAIKAN: Menggunakan metode GET yang sesuai untuk pencarian
            response = requests.post(geocoding_url, headers=headers, json=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            print(data)
            
            # Memeriksa struktur respons API yang benar
            if data and 'result' in data and data['result']:
                first_result = data['result'][0]
                coordinates[city] = {
                    'latitude': first_result['latitude'],
                    'longitude': first_result['longitude']
                }
                
                print(f"Koordinat untuk kota {city}: {coordinates[city]}")
            else:
                next
                # return {}, f"Kota '{city}' tidak dapat ditemukan. Cek nama kota atau respons API."
        except requests.exceptions.RequestException as e:
            return {}, f"Gagal terhubung ke Geocoding API: {e}"
        except (KeyError, IndexError):
            return {}, f"Struktur data API tidak sesuai untuk kota '{city}'. Pastikan Anda menggunakan API yang benar."
        except Exception as e:
            return {}, f"Error saat memproses geocoding untuk kota '{city}': {e}"
            
    return coordinates, None

def build_distance_matrix_from_coords(coordinates: Dict[str, Dict], api_key: str) -> Tuple[Dict[Tuple[str, str], float], str | None]:
    jarak_matrix = {}
    distance_url = 'https://distance-calculator.p.rapidapi.com/v1/one_to_one'
    headers = {
        "x-rapidapi-key": "9b89da684cmsh620913a3417a2ddp17984bjsn014532dfb106",
        "x-rapidapi-host": "distance-calculator.p.rapidapi.com",
        "Content-Type": "application/json"
    }
    
    # Get Jakarta's coordinates (hardcoded or from coordinates dict if available)
    jakarta_coords = {
        'latitude': -6.175394,
        'longitude': 106.827186
    }
    
    city_names = list(coordinates.keys())

    for city in city_names:
        if city.lower() == 'jakarta':
            # Distance from Jakarta to itself is 0
            jarak_matrix[('jakarta', 'jakarta')] = 0.0
            continue

        # Get destination city coordinates
        lat2, lon2 = coordinates[city]['latitude'], coordinates[city]['longitude']
        
        # Prepare query parameters for distance API
        querystring = {
            "start_point": f"({jakarta_coords['latitude']},{jakarta_coords['longitude']})",
            "end_point": f"({lat2},{lon2})",
            "unit": "kilometers",
            "decimal_places": "10"
        }
        
        try:
            response = requests.get(distance_url, headers=headers, params=querystring, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'distance' in data:
                distance_km = float(data['distance'])
                jarak_matrix[('jakarta', city)] = distance_km
                
                print(f"Jarak dari Jakarta ke {city}: {distance_km} km")
            else:
                error_detail = data.get('error', 'Distance not found in response')
                return {}, f"Gagal menghitung jarak antara Jakarta dan {city}. Detail: {error_detail}"
        except requests.exceptions.RequestException as e:
            return {}, f"Gagal terhubung ke Distance Matrix API: {e}"
        except Exception as e:
            return {}, f"Error saat memproses jarak untuk Jakarta-{city}: {e}"

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
        return self.total_profit / (self.berat + self.dimensi * 100)


@dataclass
class Truk:
    id: int
    kapasitas_berat: float
    kapasitas_dimensi: float
    konsumsi_bbm: float
    kota_asal: str = "Jakarta"


class Kromosom:
    def __init__(self, jumlah_barang: int, jumlah_truk: int):
        self.barang_ke_truk = [-1] * jumlah_barang
        self.urutan_kota_per_truk = [[] for _ in range(jumlah_truk)]
        self.loading_order_per_truk = [[] for _ in range(jumlah_truk)]
        self.fitness = 0.0
        self.total_profit = 0.0
        self.total_cost = 0.0
        self.penalti = 0.0
        self.penalti_barang_tidak_dimuat = 0.0
        self.is_valid = True

class ImprovedACOSolver:
    def __init__(self, jarak_matrix: Dict[Tuple[str, str], float],
                 alpha=1.0, beta=3.0, rho=0.6, n_ants=15, n_iterations=30):
        self.jarak_matrix, self.alpha, self.beta, self.rho = jarak_matrix, alpha, beta, rho
        self.n_ants, self.n_iterations = n_ants, n_iterations
        self.pheromone, self.best_distances = {}, {}

    def solve_tsp(self, kota_list: List[str], kota_asal: str = "Jakarta") -> Tuple[List[str], float]:
        if not kota_list: return [], 0.0
        if len(kota_list) == 1:
            jarak = self.jarak_matrix.get((kota_asal, kota_list[0]), 99999) * 2
            return kota_list, jarak
        cache_key = (kota_asal, tuple(sorted(kota_list)))
        if cache_key in self.best_distances: return self.best_distances[cache_key]
        nn_route, nn_distance = self._nearest_neighbor(kota_list, kota_asal)
        all_cities = [kota_asal] + kota_list
        for i in range(len(all_cities)):
            for j in range(i + 1, len(all_cities)):
                self.pheromone[(all_cities[i], all_cities[j])] = 1.0 / nn_distance if nn_distance > 0 else 1.0
                self.pheromone[(all_cities[j], all_cities[i])] = 1.0 / nn_distance if nn_distance > 0 else 1.0
        best_route, best_distance = nn_route, nn_distance
        for _ in range(self.n_iterations):
            routes, distances = [], []
            for _ in range(self.n_ants):
                route, unvisited, current_city, total_distance = [kota_asal], kota_list.copy(), kota_asal, 0
                while unvisited:
                    next_city = self._select_next_city(current_city, unvisited)
                    route.append(next_city)
                    total_distance += self.jarak_matrix.get((current_city, next_city), 99999)
                    current_city, unvisited = next_city, [c for c in unvisited if c != next_city]
                total_distance += self.jarak_matrix.get((current_city, kota_asal), 99999)
                routes.append(route[1:]); distances.append(total_distance)
                if total_distance < best_distance: best_distance, best_route = total_distance, route[1:]
            self._update_pheromone(routes, distances, best_route, best_distance, kota_asal)
        self.best_distances[cache_key] = (best_route, best_distance)
        return best_route, best_distance

    def _nearest_neighbor(self, kota_list: List[str], kota_asal: str) -> Tuple[List[str], float]:
        route, unvisited, current_city, total_distance = [], kota_list.copy(), kota_asal, 0
        while unvisited:
            nearest_city = min(unvisited, key=lambda city: self.jarak_matrix.get((current_city, city), 99999))
            route.append(nearest_city)
            total_distance += self.jarak_matrix.get((current_city, nearest_city), 99999)
            current_city, unvisited = nearest_city, [c for c in unvisited if c != nearest_city]
        total_distance += self.jarak_matrix.get((current_city, kota_asal), 99999)
        return route, total_distance

    def _select_next_city(self, current_city: str, unvisited: List[str]) -> str:
        probabilities = [ (self.pheromone.get((current_city, city), 1.0) ** self.alpha) * ((1.0 / self.jarak_matrix.get((current_city, city), 99999)) ** self.beta) for city in unvisited]
        total_prob = sum(probabilities)
        if total_prob > 0: return np.random.choice(unvisited, p=[p / total_prob for p in probabilities])
        return random.choice(unvisited)

    def _update_pheromone(self, routes, distances, best_route, best_distance, kota_asal):
        for key in self.pheromone: self.pheromone[key] *= (1 - self.rho)
        for i, route in enumerate(routes):
            full_route, p_deposit = [kota_asal] + route + [kota_asal], 1.0 / distances[i] if distances[i] > 0 else 1.0
            for j in range(len(full_route) - 1): self.pheromone[(full_route[j], full_route[j+1])] += p_deposit
        if best_route:
            full_best_route, elite_deposit = [kota_asal] + best_route + [kota_asal], 2.0 / best_distance if best_distance > 0 else 2.0
            for j in range(len(full_best_route) - 1): self.pheromone[(full_best_route[j], full_best_route[j+1])] += elite_deposit

class OptimizedGAACOOptimizer:
    def __init__(self, barang_list: List[Barang], truk_list: List[Truk],
                 jarak_matrix: Dict[Tuple[str, str], float], harga_bbm: float = 15000):
        self.barang_list, self.truk_list, self.jarak_matrix, self.harga_bbm = barang_list, truk_list, jarak_matrix, harga_bbm
        self.jumlah_barang, self.jumlah_truk = len(barang_list), len(truk_list)
        self.aco_solver = ImprovedACOSolver(jarak_matrix)
        self.sorted_barang_by_value = sorted(enumerate(barang_list), key=lambda x: x[1].value_density, reverse=True)

    def create_smart_kromosom(self) -> Kromosom:
        kromosom = Kromosom(self.jumlah_barang, self.jumlah_truk)
        truk_sisa_berat, truk_sisa_dimensi = [t.kapasitas_berat for t in self.truk_list], [t.kapasitas_dimensi for t in self.truk_list]
        for barang_idx, _ in self.sorted_barang_by_value:
            barang = self.barang_list[barang_idx]
            possible_trucks = []
            for truk_idx, _ in enumerate(self.truk_list):
                if barang.berat <= truk_sisa_berat[truk_idx] and barang.dimensi <= truk_sisa_dimensi[truk_idx]:
                    sisa = (truk_sisa_berat[truk_idx] - barang.berat) + (truk_sisa_dimensi[truk_idx] - barang.dimensi)
                    possible_trucks.append((sisa, truk_idx))
            if possible_trucks:
                best_truk_idx = min(possible_trucks, key=lambda x: x[0])[1]
                kromosom.barang_ke_truk[barang_idx] = best_truk_idx
                truk_sisa_berat[best_truk_idx] -= barang.berat
                truk_sisa_dimensi[best_truk_idx] -= barang.dimensi
        return kromosom

    def create_random_kromosom(self) -> Kromosom: return Kromosom(self.jumlah_barang, self.jumlah_truk)

    def evaluate_kromosom(self, kromosom: Kromosom) -> None:
        total_profit, total_cost, penalti, penalti_barang_tidak_dimuat = 0.0, 0.0, 0.0, 0.0
        kromosom.urutan_kota_per_truk, kromosom.loading_order_per_truk = [[] for _ in range(self.jumlah_truk)], [[] for _ in range(self.jumlah_truk)]
        truk_loads = {i: {'berat': 0, 'dimensi': 0, 'kota': set(), 'barang_indices': []} for i in range(self.jumlah_truk)}
        for barang_idx, truk_idx in enumerate(kromosom.barang_ke_truk):
            barang = self.barang_list[barang_idx]
            if truk_idx >= 0:
                truk_loads[truk_idx]['berat'] += barang.berat
                truk_loads[truk_idx]['dimensi'] += barang.dimensi
                truk_loads[truk_idx]['kota'].add(barang.kota_tujuan)
                truk_loads[truk_idx]['barang_indices'].append(barang_idx)
                total_profit += barang.total_profit
            else: penalti_barang_tidak_dimuat += barang.total_profit * 1.2
        for truk_idx, truk in enumerate(self.truk_list):
            load = truk_loads[truk_idx]
            if load['berat'] > truk.kapasitas_berat: penalti += 1e9
            if load['dimensi'] > truk.kapasitas_dimensi: penalti += 1e9
            if load['kota']:
                kota_list = list(load['kota'])
                best_route, total_distance = self.aco_solver.solve_tsp(kota_list, truk.kota_asal)
                kromosom.urutan_kota_per_truk[truk_idx] = best_route
                if best_route:
                    city_order_map = {city: i for i, city in enumerate(best_route)}
                    sorted_indices = sorted(load['barang_indices'], key=lambda idx: city_order_map.get(self.barang_list[idx].kota_tujuan, -1), reverse=True)
                    kromosom.loading_order_per_truk[truk_idx] = [self.barang_list[i].id for i in sorted_indices]
                biaya_bbm = total_distance * truk.konsumsi_bbm * self.harga_bbm
                total_cost += biaya_bbm
        kromosom.total_profit, kromosom.total_cost, kromosom.penalti, kromosom.penalti_barang_tidak_dimuat = total_profit, total_cost, penalti, penalti_barang_tidak_dimuat
        kromosom.is_valid, kromosom.fitness = (penalti == 0), (total_profit - total_cost - penalti - penalti_barang_tidak_dimuat)

    def optimize(self, population_size: int = 50, generations: int = 50, mutation_rate: float = 0.2, elite_size: int = 10) -> Kromosom:
        populasi = [self.create_smart_kromosom() for _ in range(int(population_size * 0.8))] + [self.create_random_kromosom() for _ in range(int(population_size * 0.2))]
        for kromosom in populasi: self.evaluate_kromosom(kromosom)
        best_fitness_ever, stagnation_counter = float('-inf'), 0
        for _ in range(generations):
            populasi.sort(key=lambda x: x.fitness, reverse=True)
            if populasi[0].fitness > best_fitness_ever: best_fitness_ever, stagnation_counter = populasi[0].fitness, 0
            else: stagnation_counter += 1
            if stagnation_counter > 10:
                for i in range(int(population_size * 0.2)): populasi[population_size - 1 - i] = self.create_smart_kromosom()
                stagnation_counter = 0
            new_population = populasi[:elite_size]
            while len(new_population) < population_size:
                parent1, parent2 = self.tournament_selection(populasi), self.tournament_selection(populasi)
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1, mutation_rate); self.mutate(child2, mutation_rate)
                self.evaluate_kromosom(child1); self.evaluate_kromosom(child2)
                new_population.extend([child1, child2])
            populasi = new_population
        populasi.sort(key=lambda x: x.fitness, reverse=True)
        return populasi[0]

    def tournament_selection(self, populasi: List[Kromosom], k: int = 3) -> Kromosom: return max(random.sample(populasi, k), key=lambda x: x.fitness)
    def crossover(self, parent1: Kromosom, parent2: Kromosom) -> Tuple[Kromosom, Kromosom]:
        child1, child2 = Kromosom(self.jumlah_barang, self.jumlah_truk), Kromosom(self.jumlah_barang, self.jumlah_truk)
        point = random.randint(1, self.jumlah_barang - 1)
        child1.barang_ke_truk, child2.barang_ke_truk = parent1.barang_ke_truk[:point] + parent2.barang_ke_truk[point:], parent2.barang_ke_truk[:point] + parent1.barang_ke_truk[point:]
        return child1, child2
    def mutate(self, kromosom: Kromosom, mutation_rate: float) -> None:
        for i in range(self.jumlah_barang):
            if random.random() < mutation_rate: kromosom.barang_ke_truk[i] = random.randint(-1, self.jumlah_truk - 1)

# Variabel global
current_data = {'barang_list': [], 'truk_list': [], 'jarak_matrix': {}, 'harga_bbm': 15000}
optimization_results = {}
optimization_thread = None
optimization_status = {'running': False, 'completed': False, 'error': None}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input_data')
def input_data():
    return render_template('input_data.html')

@app.route('/api/save_data', methods=['POST'])
def save_data():
    try:
        # Ganti dengan kunci API RapidAPI Anda yang valid
        RAPIDAPI_KEY = "9b89da684cmsh620913a3417a2ddp17984bjsn014532dfb106"
        
        if "GANTI_DENGAN" in RAPIDAPI_KEY:
             return jsonify({'status': 'error', 'message': 'RapidAPI Key belum diatur di dalam kode!'})

        data = request.json
        
        barang_list = [Barang(id=item['id'], berat=float(item['berat']), dimensi=float(item['dimensi']), kota_tujuan=item['kota_tujuan'].title(), profit_per_kg=float(item['profit_per_kg'])) for item in data['barang']]
        truk_list = [Truk(id=int(item['id']), kapasitas_berat=float(item['kapasitas_berat']), kapasitas_dimensi=float(item['kapasitas_dimensi']), konsumsi_bbm=float(item['konsumsi_bbm']), kota_asal=item.get('kota_asal', 'Jakarta').title()) for item in data['truk']]

        unique_cities = list(set(b.kota_tujuan for b in barang_list) | set(t.kota_asal for t in truk_list))

        coordinates, error_msg = get_coordinates_for_cities(unique_cities, RAPIDAPI_KEY)
        if error_msg:
            return jsonify({'status': 'error', 'message': error_msg})

        jarak_matrix, error_msg = build_distance_matrix_from_coords(coordinates, RAPIDAPI_KEY)
        if error_msg:
            return jsonify({'status': 'error', 'message': error_msg})

        current_data.update({
            'barang_list': barang_list, 'truk_list': truk_list,
            'jarak_matrix': jarak_matrix, 'harga_bbm': float(data.get('harga_bbm', 15000))
        })

        return jsonify({'status': 'success', 'message': 'Data berhasil disimpan dan jarak dihitung otomatis menggunakan RapidAPI!'})

    except Exception as e:
        print(f"Error di save_data: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/optimize')
def optimize_page():
    if not current_data['barang_list'] or not current_data['truk_list']:
        flash('Silakan masukkan data terlebih dahulu!', 'error')
        return redirect(url_for('input_data'))
    return render_template('optimize.html')

@app.route('/api/start_optimization', methods=['POST'])
def start_optimization():
    global optimization_thread, optimization_results, optimization_status
    if not current_data['barang_list'] or not current_data['truk_list']: return jsonify({'status': 'error', 'message': 'Data tidak tersedia.'})
    if optimization_status['running']: return jsonify({'status': 'error', 'message': 'Proses optimasi sedang berjalan.'})
    params = request.json or {}
    def run_optimization():
        global optimization_results, optimization_status
        try:
            optimization_status.update({'running': True, 'completed': False, 'error': None})
            optimizer = OptimizedGAACOOptimizer(current_data['barang_list'], current_data['truk_list'], current_data['jarak_matrix'], current_data['harga_bbm'])
            best_solution = optimizer.optimize(params.get('population_size', 50), params.get('generations', 50), params.get('mutation_rate', 0.2))
            results = {'fitness': best_solution.fitness, 'total_profit': best_solution.total_profit, 'total_cost': best_solution.total_cost, 'net_profit': best_solution.total_profit - best_solution.total_cost, 'is_valid': best_solution.is_valid, 'penalti': best_solution.penalti + best_solution.penalti_barang_tidak_dimuat, 'timestamp': datetime.now().isoformat()}
            truck_details, total_loaded_items, item_map = [], 0, {item.id: item for item in current_data['barang_list']}
            for truk_idx, truk in enumerate(current_data['truk_list']):
                loading_order_ids = best_solution.loading_order_per_truk[truk_idx]
                if not loading_order_ids: continue
                total_loaded_items += len(loading_order_ids)
                items_in_truck = [item_map[item_id] for item_id in loading_order_ids]
                truck_details.append({'truk_id': truk.id, 'total_weight': sum(item.berat for item in items_in_truck), 'capacity_weight': truk.kapasitas_berat, 'total_volume': sum(item.dimensi for item in items_in_truck), 'capacity_volume': truk.kapasitas_dimensi, 'items_count': len(loading_order_ids), 'profit': sum(item.total_profit for item in items_in_truck), 'route': best_solution.urutan_kota_per_truk[truk_idx], 'kota_asal': truk.kota_asal, 'items': [asdict(item) for item in items_in_truck]})
            unloaded_items = [asdict(current_data['barang_list'][i]) for i, t_idx in enumerate(best_solution.barang_ke_truk) if t_idx == -1]
            results.update({'truck_details': truck_details, 'unloaded_items': unloaded_items, 'unloaded_profit': sum(item['total_profit'] for item in unloaded_items), 'total_items': len(current_data['barang_list']), 'loaded_items': total_loaded_items})
            optimization_results = results
            optimization_status.update({'running': False, 'completed': True})
        except Exception as e:
            optimization_status.update({'running': False, 'error': str(e)})
            print(f"Error during optimization: {e}")
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
    return jsonify({'barang': [{'id': f'B{i:03}', 'berat': random.randint(50, 200), 'dimensi': round(random.uniform(0.2, 1.5), 2), 'kota_tujuan': random.choice(['Bandung', 'Surabaya', 'Yogyakarta', 'Malang', 'Solo']), 'profit_per_kg': 5000} for i in range(1, 41)], 'truk': [{'id': 1, 'kapasitas_berat': 2000, 'kapasitas_dimensi': 9.0, 'konsumsi_bbm': 0.15, 'kota_asal': 'Jakarta'}, {'id': 2, 'kapasitas_berat': 1500, 'kapasitas_dimensi': 8.0, 'konsumsi_bbm': 0.18, 'kota_asal': 'Jakarta'}, {'id': 3, 'kapasitas_berat': 1000, 'kapasitas_dimensi': 6.0, 'konsumsi_bbm': 0.12, 'kota_asal': 'Jakarta'}], 'harga_bbm': 15000})

@app.errorhandler(404)
def not_found_error(error): return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error): return render_template('500.html'), 500

if __name__ == '__main__':
    if not os.path.exists('templates'): os.makedirs('templates')
    if not os.path.exists('static'): os.makedirs('static')
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
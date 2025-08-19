import numpy as np
import random
import heapq
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx


# === PARAMETER MOBIL BOX ===
NUM_TRUCKS = 4  # Maksimum 4 mobil box per hari
TRUCK_CAPACITY = 700  # kg
TRUCK_DIMENSIONS = (200, 150, 150)  # cm (panjang, lebar, tinggi)
FUEL_COST_PER_KM = 3500  # Rp/km
FUEL_RATIO = 1 / 10  # 1 liter per 10 km

# Plate numbers for trucks
TRUCK_PLATES = ["B 1234 ABC", "B 2345 BCD", "B 3456 CDE", "B 4567 DEF"]

# === KLASIFIKASI DIMENSI BARANG ===
# Klasifikasi berdasarkan volume (cm³) - small, medium, large
DIMENSION_CLASSIFICATION = {
    "KECIL": 125000,  # < 50x50x50
    "MENENGAH": 1000000,  # < 100x100x100
    "BESAR": float('inf')  # >= 100x100x100
}

# Multiplier biaya berdasarkan dimensi
DIMENSION_PRICE_MULTIPLIER = {
    "KECIL": 1.0,
    "MENENGAH": 1.5,
    "BESAR": 2.0
}

# === Data Barang (ID, Nama, Berat, Dimensi, Kota Tujuan) ===
items = [
    (1, "TV", 15, (120, 20, 70), "Surabaya"),
    (2, "Mesin Cuci", 60, (85, 60, 90), "Jakarta"),
    (3, "Kulkas", 90, (70, 60, 90), "Bandung"),
    (4, "Sofa", 100, (180, 80, 90), "Semarang"),
    (5, "Lemari", 75, (200, 60, 50), "Malang"),
    (6, "Meja", 50, (120, 20, 70), "Yogyakarta"),
    (7, "TV", 40, (120, 20, 70), "Surabaya"),
    (8, "Mesin Cuci", 60, (85, 60, 90), "Jakarta"),
    (9, "Kulkas", 30, (70, 60, 80), "Bandung"),
    (10, "Sofa", 50, (180, 80, 90), "Semarang"),
    (11, "Lemari", 50, (200, 60, 50), "Malang"),
    (12, "Meja", 70, (120, 20, 70), "Yogyakarta"),
    (13, "Kipas Angin", 40, (60, 60, 50), "Surabaya"),
    (14, "AC", 120, (150, 70, 100), "Bandung"),
    (15, "Komputer", 30, (45, 25, 50), "Jakarta"),
    (16, "Radio", 20, (40, 20, 40), "Semarang"),
    (17, "Lampu", 10, (30, 30, 40), "Malang"),
    (18, "Lemari Besar", 100, (180, 100, 70), "Yogyakarta"),
    (19, "Kursi", 30, (100, 60, 90), "Surabaya"),
    (20, "Mixer", 35, (50, 40, 30), "Jakarta"),
    (21, "Blender", 25, (35, 35, 60), "Bandung"),
    (22, "Pompa Air", 80, (120, 60, 70), "Semarang"),
    (23, "Vase Bunga", 10, (30, 30, 50), "Malang"),
    (24, "Kursi Kantor", 60, (100, 60, 110), "Yogyakarta"),
    (25, "Meja Tulis", 50, (120, 50, 75), "Surabaya"),
    (26, "Microwave", 70, (60, 50, 40), "Jakarta"),
    (27, "Rak Sepatu", 20, (70, 30, 40), "Bandung"),
    (28, "Lemari Pakaian", 90, (180, 60, 100), "Semarang"),
    (29, "Vacuum Cleaner", 40, (30, 30, 90), "Malang"),
    (30, "Dispenser", 25, (40, 40, 80), "Yogyakarta"),
    (31, "Dispenser", 25, (40, 40, 80), "Jakarta"),
    (32, "Speaker", 20, (30, 30, 40), "Surabaya"),
    (33, "Printer", 35, (50, 40, 30), "Bandung"),
    (34, "Lukisan", 15, (100, 5, 80), "Yogyakarta"),
    (35, "Setrika", 10, (30, 20, 15), "Malang"),
    (36, "Kasur Lipat", 60, (180, 80, 30), "Semarang"),
    (37, "Remote", 8, (40, 10, 3), "Jakarta"),
    (38, "Alat Fitnes", 120, (150, 100, 120), "Bandung"),
    (39, "Gitar", 12, (110, 40, 15), "Surabaya"),
    (40, "Box Bayi", 70, (120, 80, 100), "Yogyakarta")
]

NUM_ITEMS = len(items)

# === Graph Kota dan Jarak (semua saling terhubung)
city_graph = {
    "Gudang": {"Jakarta": 100, "Bandung": 200},
    "Jakarta": {"Gudang": 100, "Semarang": 300, "Bandung": 150},
    "Bandung": {"Gudang": 200, "Jakarta": 150, "Yogyakarta": 400},
    "Semarang": {"Jakarta": 300, "Yogyakarta": 200, "Malang": 500},
    "Yogyakarta": {"Bandung": 400, "Semarang": 200, "Surabaya": 300},
    "Malang": {"Semarang": 500, "Surabaya": 150},
    "Surabaya": {"Yogyakarta": 300, "Malang": 150}
}


# === Fungsi untuk klasifikasi dimensi barang ===
def classify_item_dimension(dimensions):
    """Klasifikasi dimensi barang berdasarkan volume (kecil, menengah, besar)"""
    volume = dimensions[0] * dimensions[1] * dimensions[2]

    if volume < DIMENSION_CLASSIFICATION["KECIL"]:
        return "KECIL"
    elif volume < DIMENSION_CLASSIFICATION["MENENGAH"]:
        return "MENENGAH"
    else:
        return "BESAR"


# === Dijkstra untuk mencari jarak terpendek antara dua kota ===
def dijkstra(graph, start, end):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    queue = [(0, start)]

    while queue:
        curr_distance, curr_node = heapq.heappop(queue)
        if curr_node == end:
            return distances[end]

        if curr_distance > distances[curr_node]:
            continue

        for neighbor, weight in graph[curr_node].items():
            distance = curr_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))

    return distances[end]
def visualize_city_graph(graph, route=None):
    G = nx.Graph()

    # Tambahkan edge dan jaraknya
    for city in graph:
        for neighbor, distance in graph[city].items():
            G.add_edge(city, neighbor, weight=distance)

    pos = nx.spring_layout(G, seed=42)  # Untuk layout yang konsisten
    edge_labels = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue', font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    if route:
        # Gambar rute dengan warna berbeda
        path_edges = list(zip(route, route[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

    plt.title("Graph Kota dan Jarak Antar Kota")
    plt.savefig('city_graph.png')
    plt.show()



# === Mengoptimalkan urutan kota tujuan berdasarkan jarak terdekat ===
def optimize_city_order(graph, cities):
    if not cities:
        return []

    # Dict untuk menyimpan jarak antar kota
    distances = {}
    for city1 in cities:
        for city2 in cities:
            if city1 != city2:
                distances[(city1, city2)] = dijkstra(graph, city1, city2)

    # Tentukan urutan kota yang optimal dengan algoritma nearest neighbor
    unvisited = set(cities)
    ordered_cities = []

    # Mulai dari kota terdekat dengan gudang
    start_city = min(cities, key=lambda city: dijkstra(graph, "Gudang", city))
    ordered_cities.append(start_city)
    unvisited.remove(start_city)

    current_city = start_city
    while unvisited:
        # Pilih kota terdekat berikutnya
        next_city = min(unvisited, key=lambda city: distances.get((current_city, city), float('inf')))
        ordered_cities.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city

    return ordered_cities


# === Mencari rute terpendek untuk kumpulan kota ===
def optimize_route(graph, cities):
    if not cities:
        return [], 0

    # Optimasi urutan kota
    ordered_cities = optimize_city_order(graph, cities)

    # Hitung total jarak
    total_distance = dijkstra(graph, "Gudang", ordered_cities[0])
    route = ["Gudang", ordered_cities[0]]

    for i in range(1, len(ordered_cities)):
        distance = dijkstra(graph, ordered_cities[i - 1], ordered_cities[i])
        total_distance += distance
        route.append(ordered_cities[i])

    # Tambahkan jarak kembali ke gudang
    total_distance += dijkstra(graph, ordered_cities[-1], "Gudang")
    route.append("Gudang")  # Kembali ke gudang

    return route, total_distance


# === Kelompokkan item berdasarkan kota tujuan ===
def group_items_by_city():
    city_items = defaultdict(list)
    for idx, item in enumerate(items):
        city_items[item[4]].append(idx)
    return city_items


# === Reorganisasi barang dalam truk berdasarkan urutan kota ===
def reorganize_truck_by_city_order(truck_items):
    # Kelompokkan item berdasarkan kota
    items_by_city = defaultdict(list)
    for item_idx in truck_items:
        city = items[item_idx][4]
        items_by_city[city].append(item_idx)

    # Urutkan kota berdasarkan jarak dari gudang
    cities = list(items_by_city.keys())
    ordered_cities = optimize_city_order(city_graph, cities)

    # Susun ulang item berdasarkan urutan kota
    reorganized_items = []
    for city in ordered_cities:
        reorganized_items.extend(items_by_city[city])

    return reorganized_items


# === Hitung fitness berdasarkan formula yang diberikan ===
def calculate_fitness(solution):
    total_profit = 0
    delivered_items = set()

    for truck_idx, truck_items in enumerate(solution):
        if not truck_items:
            continue  # Truk kosong

        # Reorganisasi barang berdasarkan urutan kota
        truck_items = reorganize_truck_by_city_order(truck_items)

        # Cek kapasitas berat
        total_weight = sum(items[item_idx][2] for item_idx in truck_items)
        if total_weight > TRUCK_CAPACITY:
            return -1e9  # Penalti jika melebihi kapasitas

        # Hitung volume truk yang tersedia
        truck_volume = TRUCK_DIMENSIONS[0] * TRUCK_DIMENSIONS[1] * TRUCK_DIMENSIONS[2]
        used_volume = 0

        # Barang yang muat dalam truk berdasarkan volume
        items_that_fit = []

        # Cek dimensi setiap barang dan hitung total volume yang digunakan
        for item_idx in truck_items:
            item_dimensions = items[item_idx][3]

            # Cek apakah dimensi barang melebihi dimensi truk
            if any(dim > TRUCK_DIMENSIONS[i] for i, dim in enumerate(item_dimensions)):
                return -1e9  # Skip barang yang terlalu besar

            # Hitung volume barang
            item_volume = item_dimensions[0] * item_dimensions[1] * item_dimensions[2]

            # Cek apakah masih ada cukup ruang dalam truk
            if used_volume + item_volume <= truck_volume:
                used_volume += item_volume
                items_that_fit.append(item_idx)
                delivered_items.add(item_idx)
            # Jika tidak cukup ruang, barang tidak dimasukkan ke dalam truk
            else:
                return -1e9

        # Sesuaikan truck_items dengan barang yang benar-benar muat
        truck_items = items_that_fit

        # Tentukan kota yang akan dikunjungi berdasarkan barang yang muat
        cities = set()
        for item_idx in truck_items:
            cities.add(items[item_idx][4])

        # Temukan rute terpendek untuk pengiriman
        route, total_distance = optimize_route(city_graph, list(cities))

        # Hitung keuntungan berdasarkan formula: berat x jarak x multiplier dimensi
        delivery_profit = 0
        for item_idx in truck_items:
            weight = items[item_idx][2]
            city = items[item_idx][4]
            distance = dijkstra(city_graph, "Gudang", city)
            dimension_class = classify_item_dimension(items[item_idx][3])
            price_multiplier = DIMENSION_PRICE_MULTIPLIER[dimension_class]

            # Formula: berat x jarak x multiplier dimensi x harga dasar (100)
            item_profit = weight * distance * price_multiplier * 100
            delivery_profit += item_profit

        # Hitung biaya bahan bakar berdasarkan rute optimal
        fuel_cost = total_distance * FUEL_COST_PER_KM

        # Keuntungan truk = keuntungan pengiriman - biaya bahan bakar
        truck_profit = delivery_profit - fuel_cost
        total_profit += truck_profit

    # Hitung penalti untuk barang yang tidak terkirim
    undelivered_items = set(range(NUM_ITEMS)) - delivered_items
    penalty = 0

    for item_idx in undelivered_items:
        item = items[item_idx]
        weight = item[2]
        city = item[4]
        distance = dijkstra(city_graph, "Gudang", city)
        dimension_class = classify_item_dimension(item[3])
        price_multiplier = DIMENSION_PRICE_MULTIPLIER[dimension_class]

        # Penalti = 50% dari potensi keuntungan
        penalty += weight * distance * price_multiplier * 100

    # Kurangi profit dengan penalti
    total_profit -= penalty

    return total_profit


# === REPRESENTASI PARTICLE SWARM OPTIMIZATION ===

# === REPRESENTASI PARTICLE SWARM OPTIMIZATION ===

# --- PSO Position Representation ---
class ParticlePosition:
    def __init__(self, num_items, num_trucks):
        # Matriks probabilitas [NUM_ITEMS x (NUM_TRUCKS + 1)]
        # Kolom terakhir adalah probabilitas item tidak dimuat
        self.matrix = np.random.random((num_items, num_trucks + 1))
        # self.matrix[:, -1] *= 0.2  # turunkan kemungkinan tidak dimuat
        # Normalisasi baris agar jumlahnya 1 (distribusi probabilitas)
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        self.matrix = self.matrix / row_sums

    def to_solution(self):
        """Konversi matriks probabilitas ke format solusi diskrit."""
        solution = [[] for _ in range(NUM_TRUCKS)]

        # Untuk setiap item, tentukan ke truk mana item tersebut akan pergi
        for item_idx in range(NUM_ITEMS):
            # Dapatkan probabilitas untuk item ini
            probs = self.matrix[item_idx]
            # Pilih truk berdasarkan probabilitas tertinggi (kecuali kolom terakhir - tidak dimuat)
            truck_probs = probs[:-1]
            if max(truck_probs) > probs[-1]:  # Muat hanya jika probabilitas muat > tidak muat
                truck_idx = np.argmax(truck_probs)
                solution[truck_idx].append(item_idx)

        # Reorganisasi item di setiap truk berdasarkan urutan kota
        for truck_idx in range(NUM_TRUCKS):
            if solution[truck_idx]:
                solution[truck_idx] = reorganize_truck_by_city_order(solution[truck_idx])

        return solution

    def update(self, velocity):
        """Update posisi berdasarkan kecepatan."""
        self.matrix += velocity.matrix
        # Membatasi nilai ke [0, 1]
        self.matrix = np.clip(self.matrix, 0, 1)
        # Normalisasi baris agar jumlahnya 1
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # hindari pembagian nol
        self.matrix = self.matrix / row_sums

    def copy(self):
        """Buat salinan dalam posisi."""
        new_pos = ParticlePosition(0, 0)  # Buat posisi kosong
        new_pos.matrix = self.matrix.copy()
        return new_pos


# --- PSO Velocity Representation ---
class ParticleVelocity:
    def __init__(self, num_items, num_trucks):
        # Inisialisasi matriks kecepatan dengan nilai kecil
        self.matrix = np.random.uniform(-0.5, 0.5, (num_items, num_trucks + 1))

    def update(self, w, c1, c2, r1, r2, current_pos, pbest_pos, gbest_pos):
        """Update kecepatan menggunakan rumus PSO standar."""
        # w * v
        self.matrix *= w

        # c1 * r1 * (pbest - x)
        cognitive = c1 * r1 * (pbest_pos.matrix - current_pos.matrix)

        # c2 * r2 * (gbest - x)
        social = c2 * r2 * (gbest_pos.matrix - current_pos.matrix)

        # v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        self.matrix += cognitive + social

        # Batasi kecepatan ke [-1, 1] untuk mencegah langkah terlalu besar
        self.matrix = np.clip(self.matrix, -1, 1)

    def copy(self):
        """Buat salinan dalam kecepatan."""
        new_vel = ParticleVelocity(0, 0)  # Buat kecepatan kosong
        new_vel.matrix = self.matrix.copy()
        return new_vel


# --- Inisialisasi partikel untuk PSO ---
def initialize_particles(num_particles, num_items, num_trucks):
    particles = []
    velocities = []

    for _ in range(num_particles):
        particle = ParticlePosition(num_items, num_trucks)
        velocity = ParticleVelocity(num_items, num_trucks)
        particles.append(particle)
        velocities.append(velocity)

    return particles, velocities


# --- Implementasi algoritma PSO ---
def run_pso(num_particles, num_iterations, num_trucks, num_items, w=0.9, c1=1.5, c2=1.5):
    # Inisialisasi partikel dan kecepatan
    particles, velocities = initialize_particles(num_particles, num_items, num_trucks)

    # Inisialisasi pbest dan gbest
    pbest_positions = [particle.copy() for particle in particles]
    pbest_solutions = [particle.to_solution() for particle in particles]
    pbest_fitness = [calculate_fitness(solution) for solution in pbest_solutions]

    # Temukan gbest awal
    gbest_idx = np.argmax(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_solution = pbest_solutions[gbest_idx]
    gbest_fitness = pbest_fitness[gbest_idx]

    # History untuk melacak progres
    gbest_history = [gbest_fitness]

    print("=== OPTIMASI PSO UNTUK SISTEM EKSPEDISI ===")
    print(f"Jumlah Partikel: {num_particles}")
    print(f"Jumlah Iterasi: {num_iterations}")
    print(f"Jumlah Truk: {NUM_TRUCKS}")
    print(f"Jumlah Item: {NUM_ITEMS}")
    print("Memulai proses optimasi...")

    for iteration in range(num_iterations):
        for i in range(num_particles):
            # Generate koefisien random
            r1, r2 = random.random(), random.random()

            # Update kecepatan
            velocities[i].update(w, c1, c2, r1, r2, particles[i], pbest_positions[i], gbest_position)

            # Update posisi
            particles[i].update(velocities[i])

            # Konversi posisi ke solusi diskrit
            current_solution = particles[i].to_solution()

            # Evaluasi fitness
            current_fitness = calculate_fitness(current_solution)

            # Update pbest
            if current_fitness > pbest_fitness[i]:
                pbest_positions[i] = particles[i].copy()
                pbest_solutions[i] = current_solution
                pbest_fitness[i] = current_fitness

                # Update gbest
                if current_fitness > gbest_fitness:
                    gbest_position = particles[i].copy()
                    gbest_solution = current_solution
                    gbest_fitness = current_fitness

        # Kurangi inertia weight secara linear
        w = max(0.4, 0.9 - 0.5 * (iteration / num_iterations))

        # Simpan history
        gbest_history.append(gbest_fitness)
        print(f"Iterasi {iteration + 1}: Profit Terbaik = Rp {gbest_fitness:.2f}")
        print(gbest_solution)

        # Detail untuk 4 iterasi pertama
        # if iteration < 4:
        print(f"  Detail Solusi Iterasi {iteration + 1}:")
        for truck_idx, truck_items in enumerate(gbest_solution):
            print(f"  - Truk {truck_idx + 1} ({TRUCK_PLATES[truck_idx]}): {len(truck_items)} barang")

    print(f"\nOptimasi Selesai! Profit Terbaik: Rp {gbest_fitness:.2f}")

    return gbest_solution, gbest_fitness, gbest_history


# --- Visualisasi progres ---
def visualize_progress(gbest_history):
    plt.figure(figsize=(12, 6))

    plt.plot(gbest_history, 'b-', label='Best Fitness')

    plt.title('Perkembangan PSO untuk Sistem Ekspedisi')
    plt.xlabel('Iterasi')
    plt.ylabel('Fitness (Profit dalam Rupiah)')
    plt.legend()
    plt.grid(True)

    plt.savefig('pso_progress.png')
    plt.show()


# --- Cetak detail solusi ---
def print_solution_details(solution, fitness):
    print("\n=== SOLUSI OPTIMAL PENGIRIMAN BARANG ===")
    print(f"Total Profit: Rp {fitness:.2f}")

    # Hitung statistik untuk semua truk
    total_items_delivered = 0
    total_weight_delivered = 0
    total_distance = 0

    for truck_idx, truck_items in enumerate(solution):
        if not truck_items:
            print(f"\nTruk {truck_idx + 1} ({TRUCK_PLATES[truck_idx]}): Kosong")
            continue

        print(f"\nTruk {truck_idx + 1} ({TRUCK_PLATES[truck_idx]}):")

        # Hitung detail truk
        total_weight = sum(items[item_idx][2] for item_idx in truck_items)
        total_weight_delivered += total_weight

        # Kelompokkan barang berdasarkan kota
        cities = set(items[item_idx][4] for item_idx in truck_items)

        # Dapatkan rute optimal
        route, route_distance = optimize_route(city_graph, list(cities))
        total_distance += route_distance

        # Hitung biaya bahan bakar
        fuel_cost = route_distance * FUEL_COST_PER_KM
        fuel_volume = route_distance

        # Hitung profit pengiriman
        delivery_profit = 0
        for item_idx in truck_items:
            weight = items[item_idx][2]
            city = items[item_idx][4]
            distance = dijkstra(city_graph, "Gudang", city)
            dimension_class = classify_item_dimension(items[item_idx][3])
            price_multiplier = DIMENSION_PRICE_MULTIPLIER[dimension_class]

            item_profit = weight * distance * price_multiplier * 100
            delivery_profit += item_profit

        truck_profit = delivery_profit - fuel_cost

        print(f"Jumlah Barang: {len(truck_items)}")
        print(f"Total Berat: {total_weight} kg")
        print(f"Rute: {' -> '.join(route)}")
        print(f"Jarak Tempuh: {route_distance} km")
        print(f"Bahan Bakar: {fuel_volume:.2f} liter (Rp {fuel_cost:.2f})")
        print(f"Profit Truk: Rp {truck_profit:.2f}")

        # Detail barang
        print("Daftar Barang:")

        # Kelompokkan barang berdasarkan kota untuk tampilan yang lebih rapi
        items_by_city = defaultdict(list)
        for item_idx in truck_items:
            items_by_city[items[item_idx][4]].append(item_idx)

        # Tampilkan barang berdasarkan urutan kota
        ordered_cities = optimize_city_order(city_graph, list(items_by_city.keys()))
        for city in ordered_cities:
            print(f"  Kota {city}:")
            for item_idx in items_by_city[city]:
                item = items[item_idx]
                dimension_class = classify_item_dimension(item[3])
                print(f"    - {item[1]} (ID: {item[0]}, {item[2]} kg, {dimension_class})")

        total_items_delivered += len(truck_items)

    # Tampilkan statistik keseluruhan
    print("\n=== STATISTIK KESELURUHAN ===")
    print(f"Total Barang Terkirim: {total_items_delivered} dari {NUM_ITEMS}")
    print(f"Total Berat Terkirim: {total_weight_delivered} kg")
    print(f"Total Jarak Tempuh: {total_distance} km")
    print(f"Total Biaya Bahan Bakar: Rp {total_distance * FUEL_COST_PER_KM :.2f}")
    print(f"Total Profit: Rp {fitness:.2f}")


# Add this at the end of the visualize_item_distribution function
def visualize_item_distribution(solution):
    """Visualisasi distribusi barang antara truk."""
    truck_items_count = [len(truck_items) for truck_items in solution]
    truck_weights = [sum(items[item_idx][2] for item_idx in truck_items) if truck_items else 0
                     for truck_items in solution]
    truck_cities = [len(set(items[item_idx][4] for item_idx in truck_items)) if truck_items else 0
                    for truck_items in solution]

    # Calculate volume usage for each truck
    truck_volumes = []
    max_truck_volume = TRUCK_DIMENSIONS[0] * TRUCK_DIMENSIONS[1] * TRUCK_DIMENSIONS[2]

    for truck_items in solution:
        if not truck_items:
            truck_volumes.append(0)
            continue

        # Calculate total volume of items in the truck
        total_volume = 0
        for item_idx in truck_items:
            item_dimensions = items[item_idx][3]
            item_volume = item_dimensions[0] * item_dimensions[1] * item_dimensions[2]
            total_volume += item_volume

        truck_volumes.append(total_volume)

    # Buat figure dengan 4 subplots (tambahkan subplot untuk volume)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))

    # Plot jumlah barang per truk
    truck_labels = [f"Truk {i + 1}" for i in range(NUM_TRUCKS)]
    ax1.bar(truck_labels, truck_items_count, color='skyblue')
    ax1.set_title('Jumlah Barang per Truk')
    ax1.set_ylabel('Jumlah Barang')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Tambahkan label jumlah barang di atas bar
    for i, count in enumerate(truck_items_count):
        ax1.text(i, count + 0.5, str(count), ha='center')

    # Plot berat per truk
    ax2.bar(truck_labels, truck_weights, color='lightgreen')
    ax2.axhline(y=TRUCK_CAPACITY, color='red', linestyle='--', label=f'Kapasitas ({TRUCK_CAPACITY} kg)')
    ax2.set_title('Distribusi Berat per Truk')
    ax2.set_ylabel('Berat (kg)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.legend()

    # Tambahkan label berat di atas bar
    for i, weight in enumerate(truck_weights):
        ax2.text(i, weight + 10, f"{weight} kg", ha='center')

    # Plot jumlah kota per truk
    ax3.bar(truck_labels, truck_cities, color='salmon')
    ax3.set_title('Jumlah Kota yang Dikunjungi per Truk')
    ax3.set_ylabel('Jumlah Kota')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    # Tambahkan label jumlah kota di atas bar
    for i, city_count in enumerate(truck_cities):
        ax3.text(i, city_count + 0.2, str(city_count), ha='center')

    # Plot penggunaan volume per truk
    ax4.bar(truck_labels, truck_volumes, color='purple')
    ax4.axhline(y=max_truck_volume, color='red', linestyle='--',
                label=f'Maks Volume ({max_truck_volume / 1000000:.2f} m³)')
    ax4.set_title('Penggunaan Volume per Truk')
    ax4.set_ylabel('Volume (cm³)')
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    ax4.legend()

    # Tambahkan label volume di atas bar (dalam m³ untuk kemudahan pembacaan)
    for i, volume in enumerate(truck_volumes):
        volume_m3 = volume / 1000000  # Convert cm³ to m³
        ax4.text(i, volume + max_truck_volume * 0.05, f"{volume_m3:.2f} m³", ha='center')

    # Tambahkan persentase penggunaan volume
    for i, volume in enumerate(truck_volumes):
        percentage = (volume / max_truck_volume) * 100
        ax4.text(i, volume / 2, f"{percentage:.1f}%", ha='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig('item_distribution.png')
    plt.show()


# --- Main function untuk menjalankan optimasi ---
def main():
    # Parameter PSO
    num_particles = 10
    num_iterations = 30

    # Jalankan PSO
    best_solution, best_fitness, gbest_history = run_pso(
        num_particles, num_iterations, NUM_TRUCKS, NUM_ITEMS
    )

    # Tampilkan hasil
    print_solution_details(best_solution, best_fitness)

    # Visualisasi
    visualize_progress(gbest_history)
    visualize_item_distribution(best_solution)


if __name__ == "__main__":
    main()
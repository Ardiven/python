import numpy as np
import random
import heapq
import copy
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Parameter Mobil Box ---
NUM_TRUCKS = 4
TRUCK_CAPACITY = 700  # kg
TRUCK_DIMENSIONS = (200, 150, 150)  # cm (panjang, lebar, tinggi)
FUEL_COST_PER_KM = 3500  # Rp/km
PRICE_PER_KG = 100  # Harga per kg
USE_ORDER_OPTIMIZATION = False  # Toggle untuk aktifkan/disable optimize_city_order


# --- Data Barang (ID, Nama, Berat, Dimensi, Kota Tujuan) ---
items = [
    (1, "TV", 15, (120, 20, 70), "Surabaya"),
    (2, "Mesin Cuci", 60, (85, 60, 90), "Jakarta"),
    (3, "Kulkas", 90, (70, 60, 90), "Bandung"),
    (4, "Sofa", 100, (180, 80, 90), "Semarang"),
    (5, "Lemari", 75, (200, 60, 50), "Malang"),
    (6, "Meja", 50, (120, 20, 70), "Yogyakarta"),
    (7, "TV", 400, (120, 20, 70), "Surabaya"),
    (8, "Mesin Cuci", 160, (85, 60, 90), "Jakarta"),
    (9, "Kulkas", 30, (70, 60, 80), "Bandung"),
    (10, "Sofa", 50, (180, 80, 90), "Semarang"),
    (11, "Lemari", 50, (200, 60, 50), "Malang"),
    (12, "Meja", 70, (120, 20, 70), "Yogyakarta"),
    (13, "Kipas Angin", 40, (60, 60, 50), "Surabaya"),
    (14, "AC", 120, (150, 70, 100), "Bandung"),
    (15, "Komputer", 30, (45, 25, 50), "Jakarta"),
    (16, "Radio", 20, (40, 20, 40), "Semarang"),
    (17, "Lampu", 10, (30, 30, 40), "Malang"),
    (18, "Peti Mati", 200, (180, 100, 70), "Yogyakarta"),
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
    (30, "Helikopter Mainan", 5, (40, 40, 25), "Yogyakarta"),
    (31, "Dispenser", 25, (40, 40, 80), "Jakarta"),
    (32, "Speaker", 20, (30, 30, 40), "Surabaya"),
    (33, "Printer", 35, (50, 40, 30), "Bandung"),
    (34, "Lukisan", 15, (100, 5, 80), "Yogyakarta"),
    (35, "Setrika", 10, (30, 20, 15), "Malang"),
    (36, "Kasur Lipat", 60, (180, 80, 30), "Semarang"),
    (37, "Drone", 8, (40, 40, 20), "Jakarta"),
    (38, "Alat Fitnes", 120, (150, 100, 120), "Bandung"),
    (39, "Gitar", 12, (110, 40, 15), "Surabaya"),
    (40, "Box Bayi", 70, (120, 80, 100), "Yogyakarta")
]

# --- Graph Kota dan Jarak (semua saling terhubung)
city_graph = {
    "Gudang": {"Jakarta": 100, "Bandung": 200},
    "Jakarta": {"Gudang": 100, "Semarang": 300, "Bandung": 150},
    "Bandung": {"Gudang": 200, "Jakarta": 150, "Yogyakarta": 400},
    "Semarang": {"Jakarta": 300, "Yogyakarta": 200, "Malang": 500},
    "Yogyakarta": {"Bandung": 400, "Semarang": 200, "Surabaya": 300},
    "Malang": {"Semarang": 500, "Surabaya": 150},
    "Surabaya": {"Yogyakarta": 300, "Malang": 150}
}

NUM_ITEMS = len(items)


# --- Dijkstra untuk jarak dari kota start ke end ---
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


# --- Mengoptimalkan urutan kota tujuan berdasarkan jarak terdekat ---
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


# --- Mencari rute terpendek untuk kumpulan kota ---
def optimize_route(graph, cities):
    if not cities:
        return [], 0

    # Urutkan kota secara optimal atau tidak
    if USE_ORDER_OPTIMIZATION:
        ordered_cities = optimize_city_order(graph, cities)
    else:
        ordered_cities = list(cities)

    total_distance = dijkstra(graph, "Gudang", ordered_cities[0])
    route = ["Gudang", ordered_cities[0]]

    for i in range(1, len(ordered_cities)):
        distance = dijkstra(graph, ordered_cities[i - 1], ordered_cities[i])
        total_distance += distance
        route.append(ordered_cities[i])

    total_distance += dijkstra(graph, ordered_cities[-1], "Gudang")
    route.append("Gudang")

    return route, total_distance



# --- Kelompokkan item berdasarkan kota tujuan ---
def group_items_by_city():
    city_items = defaultdict(list)
    for idx, item in enumerate(items):
        city_items[item[4]].append(idx)
    return city_items


# --- Reorganisasi barang dalam truk berdasarkan urutan kota ---
def reorganize_truck_by_city_order(truck_items):
    # Kelompokkan item berdasarkan kota
    items_by_city = defaultdict(list)
    for item_idx in truck_items:
        city = items[item_idx][4]
        items_by_city[city].append(item_idx)

    cities = list(items_by_city.keys())

    # Tentukan urutan kota
    if USE_ORDER_OPTIMIZATION:
        ordered_cities = optimize_city_order(city_graph, cities)
    else:
        ordered_cities = cities  # urutan asli
        # Atau random.shuffle(cities) jika ingin eksplorasi acak

    # Susun ulang item berdasarkan urutan kota
    reorganized_items = []
    for city in ordered_cities:
        reorganized_items.extend(items_by_city[city])

    return reorganized_items


def calculate_sequential_route(graph, truck_items):
    if not truck_items:
        return [], 0

    route = ["Gudang"]
    total_distance = 0
    current_city = "Gudang"

    for item_idx in truck_items:
        next_city = items[item_idx][4]
        distance = dijkstra(graph, current_city, next_city)
        total_distance += distance
        route.append(next_city)
        current_city = next_city

    return route, total_distance

# --- Hitung fitness berdasarkan rute pengantaran dengan mempertimbangkan urutan ---
def calculate_fitness_with_order(solution):
    total_profit = 0

    # Set to track delivered items
    delivered_items = set()

    for truck_idx, truck_items in enumerate(solution):
        if not truck_items:
            continue  # Empty truck, no profit or cost

        # Reorganisasi barang berdasarkan urutan kota
        truck_items = reorganize_truck_by_city_order(truck_items)

        # Check weight capacity
        total_weight = sum(items[item_idx][2] for item_idx in truck_items)
        if total_weight > TRUCK_CAPACITY:
            return -1e9  # Penalty if exceeding capacity

        # Calculate available truck volume
        truck_volume = TRUCK_DIMENSIONS[0] * TRUCK_DIMENSIONS[1] * TRUCK_DIMENSIONS[2]
        used_volume = 0

        # Track which items can actually fit in the truck
        items_that_fit = []

        # Check dimensions of each item and calculate total used volume
        for item_idx in truck_items:
            item_dimensions = items[item_idx][3]

            # Check if item dimensions exceed truck dimensions
            if any(dim > TRUCK_DIMENSIONS[i] for i, dim in enumerate(item_dimensions)):
                return -1e9  # Penalty if too large

            # Calculate item volume
            item_volume = item_dimensions[0] * item_dimensions[1] * item_dimensions[2]

            # Check if there's enough space left in the truck
            if used_volume + item_volume <= truck_volume:
                used_volume += item_volume
                items_that_fit.append(item_idx)
                delivered_items.add(item_idx)
            # If item doesn't fit, it's not added to delivered_items

        # If some items don't fit, adjust truck_items to only include those that fit
        truck_items = items_that_fit

        # Determine cities to visit based on items that fit
        cities = []
        items_by_city = {}

        for item_idx in truck_items:
            city = items[item_idx][4]
            weight = items[item_idx][2]

            if city not in cities:
                cities.append(city)
                items_by_city[city] = weight
            else:
                items_by_city[city] += weight

        # Find shortest route for delivery
        # Hitung rute sesuai urutan barang
        route, total_distance = calculate_sequential_route(city_graph, truck_items)

        # Calculate profit from goods
        delivery_profit = sum(items_by_city[city] * PRICE_PER_KG * dijkstra(city_graph, "Gudang", city)
                              for city in cities)

        # Calculate fuel cost based on optimal route
        fuel_cost = total_distance * FUEL_COST_PER_KM

        # Bonus untuk barang kota yang sama yang dikelompokkan bersama
        city_grouping_bonus = 0
        current_city = None
        consecutive_same_city = 0

        for item_idx in truck_items:
            city = items[item_idx][4]
            if city == current_city:
                consecutive_same_city += 1
                city_grouping_bonus += consecutive_same_city * 1000  # Bonus meningkat untuk pengelompokan lebih banyak
            else:
                current_city = city
                consecutive_same_city = 0

        truck_profit = delivery_profit - fuel_cost + city_grouping_bonus
        total_profit += truck_profit

    # Calculate penalty for items not delivered
    undelivered_items = set(range(NUM_ITEMS)) - delivered_items

    for item_idx in undelivered_items:
        item = items[item_idx]
        weight = item[2]
        city = item[4]
        distance = dijkstra(city_graph, "Gudang", city)

        # Penalty = Weight x Price per kg x Distance
        penalty = weight * PRICE_PER_KG * distance
        total_profit -= penalty  # Reduce profit by penalty

    return total_profit


# --- PSO Position Representation for the Truck Loading Problem ---
# Each particle is represented as a probability matrix of size [NUM_ITEMS x (NUM_TRUCKS + 1)]
# The last column represents the probability of an item not being loaded
class ParticlePosition:
    def __init__(self, num_items, num_trucks):
        self.matrix = np.random.random((num_items, num_trucks + 1))
        # Normalize rows to sum to 1 (probability distribution)
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        self.matrix = self.matrix / row_sums

    def to_solution(self):
        """Convert probability matrix to a discrete solution format."""
        solution = [[] for _ in range(NUM_TRUCKS)]

        # For each item, decide which truck it goes to
        for item_idx in range(NUM_ITEMS):
            # Get probabilities for this item
            probs = self.matrix[item_idx]
            # Choose truck based on highest probability (excluding the last column - not loaded)
            truck_probs = probs[:-1]
            if max(truck_probs) > probs[-1]:  # Only load if probability of loading > not loading
                truck_idx = np.argmax(truck_probs)
                solution[truck_idx].append(item_idx)

        # Reorganize items in each truck by city order
        for truck_idx in range(NUM_TRUCKS):
            if solution[truck_idx]:
                solution[truck_idx] = reorganize_truck_by_city_order(solution[truck_idx])

        return solution

    def update(self, velocity):
        """Update position based on velocity."""
        self.matrix += velocity.matrix
        # Clip values to [0, 1]
        self.matrix = np.clip(self.matrix, 0, 1)
        # Normalize rows to sum to 1
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        self.matrix = self.matrix / row_sums

    def copy(self):
        """Create a deep copy of position."""
        new_pos = ParticlePosition(0, 0)  # Create empty position
        new_pos.matrix = self.matrix.copy()
        return new_pos


# --- PSO Velocity Representation ---
class ParticleVelocity:
    def __init__(self, num_items, num_trucks):
        # Initialize velocity matrix with small values
        self.matrix = np.random.uniform(-0.1, 0.1, (num_items, num_trucks + 1))

    def update(self, w, c1, c2, r1, r2, current_pos, pbest_pos, gbest_pos):
        """Update velocity using standard PSO formula."""
        # w * v
        self.matrix *= w

        # c1 * r1 * (pbest - x)
        cognitive = c1 * r1 * (pbest_pos.matrix - current_pos.matrix)

        # c2 * r2 * (gbest - x)
        social = c2 * r2 * (gbest_pos.matrix - current_pos.matrix)

        # v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        self.matrix += cognitive + social

        # Limit velocity to [-1, 1] to prevent too large steps
        self.matrix = np.clip(self.matrix, -1, 1)

    def copy(self):
        """Create a deep copy of velocity."""
        new_vel = ParticleVelocity(0, 0)  # Create empty velocity
        new_vel.matrix = self.matrix.copy()
        return new_vel


# --- Initialize particles for pure PSO ---
def initialize_particles(num_particles, num_items, num_trucks):
    particles = []
    velocities = []

    for _ in range(num_particles):
        particle = ParticlePosition(num_items, num_trucks)
        velocity = ParticleVelocity(num_items, num_trucks)
        particles.append(particle)
        velocities.append(velocity)

    return particles, velocities


# --- Pure PSO algorithm implementation ---
def pure_pso(num_particles, num_iterations, num_trucks, num_items, w=0.5, c1=1.5, c2=1.5):
    # Initialize particles and velocities
    particles, velocities = initialize_particles(num_particles, num_items, num_trucks)

    # Initialize pbest and gbest
    pbest_positions = [particle.copy() for particle in particles]
    pbest_solutions = [particle.to_solution() for particle in particles]
    pbest_fitness = [calculate_fitness_with_order(solution) for solution in pbest_solutions]

    # Find initial gbest
    gbest_idx = np.argmax(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_solution = pbest_solutions[gbest_idx]
    gbest_fitness = pbest_fitness[gbest_idx]

    # History for tracking progress
    gbest_history = [gbest_fitness]
    avg_fitness_history = [sum(pbest_fitness) / len(pbest_fitness)]

    print("Starting Pure PSO Optimization (Ordered Delivery)...")

    for iteration in range(num_iterations):
        for i in range(num_particles):
            # Generate random coefficients
            r1, r2 = random.random(), random.random()

            # Update velocity
            velocities[i].update(w, c1, c2, r1, r2, particles[i], pbest_positions[i], gbest_position)

            # Update position
            particles[i].update(velocities[i])

            # Convert position to discrete solution
            current_solution = particles[i].to_solution()

            # Evaluate fitness
            current_fitness = calculate_fitness_with_order(current_solution)

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

        # Decrease inertia weight linearly
        w = 0.9 - 0.5 * (iteration / num_iterations)

        # Save history
        gbest_history.append(gbest_fitness)
        avg_fitness_history.append(sum(pbest_fitness) / len(pbest_fitness))

        print(f"Iterasi {iteration + 1}: Profit Terbaik = {gbest_fitness}")
        print(gbest_solution)

    return gbest_solution, gbest_fitness, gbest_history, avg_fitness_history


# --- Visualize progress ---
def visualize_progress(gbest_history, avg_fitness_history):
    plt.figure(figsize=(12, 6))
    plt.plot(gbest_history, label='Best Fitness')
    plt.title('Pure PSO Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (Profit)')
    plt.legend()
    plt.grid(True)
    plt.savefig('pso_progress.png')
    plt.show()

def visualize_delivered_vs_undelivered(delivered_items: set):
    """Visualisasi barang terkirim vs tidak."""
    total = NUM_ITEMS
    delivered_count = len(delivered_items)
    undelivered_count = total - delivered_count

    labels = ['Terkirim', 'Tidak Terkirim']
    counts = [delivered_count, undelivered_count]
    colors = ['green', 'red']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Bar chart
    ax1.bar(labels, counts, color=colors)
    ax1.set_title('Jumlah Barang Terkirim vs Tidak Terkirim')
    ax1.set_ylabel('Jumlah Barang')
    for i, count in enumerate(counts):
        ax1.text(i, count + 0.5, str(count), ha='center')

    # Pie chart
    ax2.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Persentase Pengiriman Barang')

    plt.tight_layout()
    plt.savefig('delivered_vs_undelivered.png')
    plt.show()

    # Daftar barang tidak terkirim (opsional)
    print("\n--- Barang Tidak Terkirim ---")
    undelivered = set(range(NUM_ITEMS)) - delivered_items
    if not undelivered:
        print("Semua barang berhasil dikirim! 🎉")
    else:
        for idx in undelivered:
            item = items[idx]
            print(f"- {item[1]} ({item[2]} kg) ke {item[4]}")


# --- Print solution details ---
def print_solution_details(solution, fitness):
    delivered_items = set()
    print("\n=== SOLUSI OPTIMAL ===")
    print(f"Total Profit: Rp {fitness}")

    for truck_idx, truck_items in enumerate(solution):
        if not truck_items:
            print(f"\nTruk {truck_idx + 1}: Kosong")
            continue

        print(f"\nTruk {truck_idx + 1}:")

        # Calculate truck details
        total_weight = sum(items[item_idx][2] for item_idx in truck_items)

        # Group items by city
        cities = set(items[item_idx][4] for item_idx in truck_items)

        # Get optimal route
        # Get sequential route
        route, total_distance = calculate_sequential_route(city_graph, truck_items)

        print(f"Jumlah Barang: {len(truck_items)}")
        print(f"Total Berat: {total_weight} kg")
        print(f"Rute: {' -> '.join(route)}")
        print(f"Jarak Tempuh: {total_distance} km")
        print(f"Biaya Bahan Bakar: Rp {total_distance * FUEL_COST_PER_KM}")

        # Items detail
        print("Daftar Barang:")
        for item_idx in truck_items:
            delivered_items.add(item_idx)
            item = items[item_idx]
            print(f"  - {item[1]} ({item[2]} kg) ke {item[4]}")
    # Visualisasi tambahan
    visualize_delivered_vs_undelivered(delivered_items)

# --- Main Program ---
# Parameter
NUM_PARTICLES = 20
NUM_ITERATIONS = 50
w = 0.9  # Initial inertia weight
c1, c2 = 1.5, 1.5  # Cognitive and social coefficients

# Run pure PSO algorithm
best_solution, best_fitness, gbest_history, avg_fitness_history = pure_pso(
    NUM_PARTICLES, NUM_ITERATIONS, NUM_TRUCKS, NUM_ITEMS, w, c1, c2)

# Visualize progress
progress_plot = visualize_progress(gbest_history, avg_fitness_history)

# Print solution details
print_solution_details(best_solution, best_fitness)


# --- Additional visualization functions ---
def visualize_item_distribution(solution):
    """Visualize distribution of items among trucks."""
    truck_items_count = [len(truck_items) for truck_items in solution]
    truck_weights = [sum(items[item_idx][2] for item_idx in truck_items) if truck_items else 0
                     for truck_items in solution]
    truck_cities = [len(set(items[item_idx][4] for item_idx in truck_items)) if truck_items else 0
                    for truck_items in solution]

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot number of items per truck
    truck_labels = [f"Truck {i + 1}" for i in range(NUM_TRUCKS)]
    ax1.bar(truck_labels, truck_items_count, color='skyblue')
    ax1.set_title('Number of Items per Truck')
    ax1.set_ylabel('Number of Items')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Add item count labels on top of bars
    for i, count in enumerate(truck_items_count):
        ax1.text(i, count + 0.5, str(count), ha='center')

    # Plot weight per truck
    ax2.bar(truck_labels, truck_weights, color='lightgreen')
    ax2.axhline(y=TRUCK_CAPACITY, color='red', linestyle='--', label=f'Capacity ({TRUCK_CAPACITY} kg)')
    ax2.set_title('Weight Distribution per Truck')
    ax2.set_ylabel('Weight (kg)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.legend()

    # Add weight labels on top of bars
    for i, weight in enumerate(truck_weights):
        ax2.text(i, weight + 20, f"{weight} kg", ha='center')

    # Plot number of cities per truck
    ax3.bar(truck_labels, truck_cities, color='salmon')
    ax3.set_title('Number of Cities per Truck')
    ax3.set_ylabel('Number of Cities')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    # Add city count labels on top of bars
    for i, count in enumerate(truck_cities):
        ax3.text(i, count + 0.1, str(count), ha='center')

    plt.tight_layout()
    plt.savefig('item_distribution.png')
    plt.show()


def visualize_truck_volume_usage(solution):
    """Visualize volume usage in each truck."""
    total_truck_volume = TRUCK_DIMENSIONS[0] * TRUCK_DIMENSIONS[1] * TRUCK_DIMENSIONS[2]

    # Calculate used volume for each truck
    used_volumes = []
    for truck_items in solution:
        if not truck_items:
            used_volumes.append(0)
            continue

        used_volume = sum(items[item_idx][3][0] * items[item_idx][3][1] * items[item_idx][3][2]
                          for item_idx in truck_items)
        used_volumes.append(used_volume)

    # Calculate percentage of volume used
    volume_percentages = [(vol / total_truck_volume) * 100 for vol in used_volumes]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot volume used vs total volume
    truck_labels = [f"Truck {i + 1}" for i in range(NUM_TRUCKS)]

    # Bar chart with used and unused volume
    bar_width = 0.35
    unused_volumes = [total_truck_volume - vol for vol in used_volumes]

    ax1.bar(truck_labels, used_volumes, bar_width, label='Used Volume (cm³)', color='cornflowerblue')
    ax1.bar(truck_labels, unused_volumes, bar_width, bottom=used_volumes,
            label='Available Volume (cm³)', color='lightgray', alpha=0.7)

    ax1.set_title('Volume Usage per Truck')
    ax1.set_ylabel('Volume (cm³)')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Add volume labels on bars
    for i, vol in enumerate(used_volumes):
        if vol > 0:
            ax1.text(i, vol / 2, f"{int(vol):,}", ha='center', va='center')

    # Pie charts for volume usage
    colors = ['cornflowerblue', 'lightgray']

    ax2.set_title('Percentage of Volume Used per Truck')

    # Create mini pie charts for each truck
    for i in range(NUM_TRUCKS):
        # Add a small subplot for each pie chart
        ax_pie = fig.add_subplot(1, NUM_TRUCKS * 2, NUM_TRUCKS + i + 1)
        used = min(max(volume_percentages[i], 0), 100)
        unused = 100 - used
        percentages = [used, unused]

        ax_pie.pie(percentages, colors=colors, autopct='%1.1f%%', startangle=90)
        ax_pie.set_title(f"Truck {i + 1}")

    plt.tight_layout()
    plt.savefig('truck_volume_usage.png')
    plt.show()


def visualize_city_distribution(solution):
    """Visualize distribution of cities among trucks."""
    # Count items per city per truck
    city_distribution = {}

    for truck_idx, truck_items in enumerate(solution):
        if not truck_items:
            continue

        truck_name = f"Truck {truck_idx + 1}"

        for item_idx in truck_items:
            city = items[item_idx][4]

            if city not in city_distribution:
                city_distribution[city] = {truck_name: 1}
            elif truck_name not in city_distribution[city]:
                city_distribution[city][truck_name] = 1
            else:
                city_distribution[city][truck_name] += 1

    # Create a stacked bar chart
    cities = list(city_distribution.keys())
    truck_labels = [f"Truck {i + 1}" for i in range(NUM_TRUCKS)]

    # Prepare data for plotting
    data = np.zeros((len(cities), NUM_TRUCKS))

    for i, city in enumerate(cities):
        for j in range(NUM_TRUCKS):
            truck_name = f"Truck {j + 1}"
            if truck_name in city_distribution[city]:
                data[i, j] = city_distribution[city][truck_name]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    bottom = np.zeros(len(cities))

    for j in range(NUM_TRUCKS):
        ax.bar(cities, data[:, j], bottom=bottom, label=f'Truck {j + 1}')
        bottom += data[:, j]

    ax.set_title('Distribution of Items by City and Truck')
    ax.set_xlabel('City')
    ax.set_ylabel('Number of Items')
    ax.legend()

    plt.tight_layout()
    plt.savefig('city_distribution.png')
    plt.show()



# --- Update main program to include visualizations ---
def visualize_all_solution_aspects(solution):
    """Run all visualizations for the solution."""
    visualize_item_distribution(solution)
    visualize_truck_volume_usage(solution)
    visualize_city_distribution(solution)


# --- Add this at the end of your main program ---
# Visualize solution details
visualize_all_solution_aspects(best_solution)
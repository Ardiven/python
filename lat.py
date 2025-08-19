import numpy as np
import random
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Barang:
    id: str
    nama: str
    berat: float  # kg
    panjang: float  # cm
    lebar: float  # cm
    tinggi: float  # cm
    kota_tujuan: str
    rapuh: bool = False
    prioritas: int = 1  # 1=normal, 2=urgent

    @property
    def volume(self) -> float:
        return self.panjang * self.lebar * self.tinggi


@dataclass
class Mobil:
    id: str
    plat_nomor: str
    kapasitas_berat: float  # kg
    panjang_box: float  # cm
    lebar_box: float  # cm
    tinggi_box: float  # cm
    biaya_per_km: float = 5000  # rupiah

    @property
    def volume_box(self) -> float:
        return self.panjang_box * self.lebar_box * self.tinggi_box


@dataclass
class PosisiBarang:
    barang: Barang
    x: float
    y: float
    z: float
    rotated: bool = False  # apakah barang diputar


class PreferensiZona(Enum):
    BAWAH = "bawah"
    TENGAH = "tengah"
    ATAS = "atas"
    DEKAT_PINTU = "dekat_pintu"
    BEBAS = "bebas"


# =============================================================================
# FUZZY LOGIC MODULE
# =============================================================================

class FuzzyLogicEngine:
    def __init__(self):
        self.rules = []

    def fuzzy_berat(self, berat: float) -> Dict[str, float]:
        """Fuzzy set untuk berat: ringan, sedang, berat"""
        ringan = max(0, min(1, (10 - berat) / 10))
        sedang = max(0, min(1, (berat - 5) / 10)) if berat <= 15 else max(0, min(1, (25 - berat) / 10))
        berat_val = max(0, min(1, (berat - 15) / 15))
        return {"ringan": ringan, "sedang": sedang, "berat": berat_val}

    def fuzzy_volume(self, volume: float) -> Dict[str, float]:
        """Fuzzy set untuk volume: kecil, sedang, besar"""
        # Asumsi volume dalam cm³
        kecil = max(0, min(1, (10000 - volume) / 10000))
        sedang = max(0, min(1, (volume - 5000) / 15000)) if volume <= 20000 else max(0,
                                                                                     min(1, (35000 - volume) / 15000))
        besar = max(0, min(1, (volume - 20000) / 20000))
        return {"kecil": kecil, "sedang": sedang, "besar": besar}

    def hitung_preferensi(self, barang: Barang) -> Dict[PreferensiZona, float]:
        """Menghitung preferensi penempatan barang menggunakan fuzzy logic"""
        berat_fuzzy = self.fuzzy_berat(barang.berat)
        volume_fuzzy = self.fuzzy_volume(barang.volume)

        preferensi = {zona: 0.0 for zona in PreferensiZona}

        # Rule 1: Barang berat dan tidak rapuh -> BAWAH
        if berat_fuzzy["berat"] > 0.5 and not barang.rapuh:
            preferensi[PreferensiZona.BAWAH] += 0.9

        # Rule 2: Barang ringan dan rapuh -> ATAS
        if berat_fuzzy["ringan"] > 0.5 and barang.rapuh:
            preferensi[PreferensiZona.ATAS] += 0.8

        # Rule 3: Barang kecil -> BEBAS
        if volume_fuzzy["kecil"] > 0.5:
            preferensi[PreferensiZona.BEBAS] += 0.6

        # Rule 4: Prioritas tinggi -> DEKAT_PINTU
        if barang.prioritas >= 2:
            preferensi[PreferensiZona.DEKAT_PINTU] += 0.7

        # Normalisasi
        total = sum(preferensi.values())
        if total > 0:
            for zona in preferensi:
                preferensi[zona] /= total
        else:
            # Default equal probability jika tidak ada rule yang triggered
            for zona in preferensi:
                preferensi[zona] = 1.0 / len(PreferensiZona)

        return preferensi


# =============================================================================
# SIMULATED ANNEALING MODULE
# =============================================================================

class SimulatedAnnealing:
    def __init__(self, mobil: Mobil, barang_list: List[Barang], fuzzy_engine: FuzzyLogicEngine):
        self.mobil = mobil
        self.barang_list = barang_list
        self.fuzzy_engine = fuzzy_engine
        self.suhu_awal = 1000
        self.suhu_min = 0.1
        self.cooling_rate = 0.95
        self.max_iterasi = 500  # Reduced for faster execution

    def generate_initial_solution(self) -> List[PosisiBarang]:
        """Generate solusi awal berdasarkan heuristik fuzzy"""
        if not self.barang_list:
            return []

        solusi = []
        x_current, y_current, z_current = 0, 0, 0
        layer_height = 0

        # Urutkan barang berdasarkan prioritas dan berat
        barang_sorted = sorted(self.barang_list,
                               key=lambda b: (b.prioritas, b.berat),
                               reverse=True)

        for barang in barang_sorted:
            placed = False

            # Coba beberapa orientasi
            for rotated in [False, True]:
                panjang = barang.panjang if not rotated else barang.lebar
                lebar = barang.lebar if not rotated else barang.panjang
                tinggi = barang.tinggi

                if self._check_fit_with_dimensions(panjang, lebar, tinggi, x_current, y_current, z_current):
                    solusi.append(PosisiBarang(barang, x_current, y_current, z_current, rotated))
                    x_current += panjang
                    layer_height = max(layer_height, tinggi)
                    placed = True
                    break

                # Coba baris berikutnya
                if x_current > 0:
                    next_y = y_current + layer_height
                    if self._check_fit_with_dimensions(panjang, lebar, tinggi, 0, next_y, z_current):
                        y_current = next_y
                        x_current = 0
                        layer_height = tinggi
                        solusi.append(PosisiBarang(barang, x_current, y_current, z_current, rotated))
                        x_current += panjang
                        placed = True
                        break

            # Coba layer baru jika masih tidak muat
            if not placed:
                next_z = z_current + layer_height
                if next_z < self.mobil.tinggi_box:
                    y_current = 0
                    x_current = 0
                    z_current = next_z
                    layer_height = 0

                    for rotated in [False, True]:
                        panjang = barang.panjang if not rotated else barang.lebar
                        lebar = barang.lebar if not rotated else barang.panjang
                        tinggi = barang.tinggi

                        if self._check_fit_with_dimensions(panjang, lebar, tinggi, x_current, y_current, z_current):
                            solusi.append(PosisiBarang(barang, x_current, y_current, z_current, rotated))
                            x_current += panjang
                            layer_height = tinggi
                            placed = True
                            break

                if not placed:
                    print(f"⚠️  Barang {barang.nama} tidak dapat ditempatkan di mobil {self.mobil.plat_nomor}")

        return solusi

    def _check_fit_with_dimensions(self, panjang: float, lebar: float, tinggi: float, x: float, y: float,
                                   z: float) -> bool:
        """Cek apakah dimensi tertentu muat di posisi tertentu"""
        return (x + panjang <= self.mobil.panjang_box and
                y + lebar <= self.mobil.lebar_box and
                z + tinggi <= self.mobil.tinggi_box)

    def evaluate_fitness(self, solusi: List[PosisiBarang]) -> float:
        """Evaluate fitness dari solusi"""
        if not solusi:
            return 0

        score = 0

        # 1. Volume efficiency
        total_volume_used = sum([pos.barang.volume for pos in solusi])
        if self.mobil.volume_box > 0:
            volume_efficiency = total_volume_used / self.mobil.volume_box
            score += volume_efficiency * 100

        # 2. Penalty untuk overlap
        overlap_penalty = 0
        for i, pos1 in enumerate(solusi):
            for pos2 in solusi[i + 1:]:
                if self._check_overlap(pos1, pos2):
                    overlap_penalty += 50
        score -= overlap_penalty

        # 3. Bonus stabilitas
        stability_bonus = 0
        for pos in solusi:
            if pos.barang.berat > 15 and pos.z < self.mobil.tinggi_box * 0.3:
                stability_bonus += 10
        score += stability_bonus

        # 4. Bonus untuk barang rapuh di atas
        fragile_bonus = 0
        for pos in solusi:
            if pos.barang.rapuh and pos.z > self.mobil.tinggi_box * 0.6:
                fragile_bonus += 5
        score += fragile_bonus

        return score

    def _check_overlap(self, pos1: PosisiBarang, pos2: PosisiBarang) -> bool:
        """Check overlap dengan mempertimbangkan rotasi"""
        p1_panjang = pos1.barang.panjang if not pos1.rotated else pos1.barang.lebar
        p1_lebar = pos1.barang.lebar if not pos1.rotated else pos1.barang.panjang

        p2_panjang = pos2.barang.panjang if not pos2.rotated else pos2.barang.lebar
        p2_lebar = pos2.barang.lebar if not pos2.rotated else pos2.barang.panjang

        return not (pos1.x + p1_panjang <= pos2.x or
                    pos2.x + p2_panjang <= pos1.x or
                    pos1.y + p1_lebar <= pos2.y or
                    pos2.y + p2_lebar <= pos1.y or
                    pos1.z + pos1.barang.tinggi <= pos2.z or
                    pos2.z + pos2.barang.tinggi <= pos1.z)

    def mutate_solution(self, solusi: List[PosisiBarang]) -> List[PosisiBarang]:
        """Mutasi solusi untuk SA"""
        if not solusi or len(solusi) < 2:
            return solusi[:]

        new_solusi = [PosisiBarang(pos.barang, pos.x, pos.y, pos.z, pos.rotated) for pos in solusi]

        mutation_type = random.choice(['swap', 'move', 'rotate'])

        if mutation_type == 'swap' and len(new_solusi) >= 2:
            i, j = random.sample(range(len(new_solusi)), 2)
            new_solusi[i].x, new_solusi[j].x = new_solusi[j].x, new_solusi[i].x
            new_solusi[i].y, new_solusi[j].y = new_solusi[j].y, new_solusi[i].y
            new_solusi[i].z, new_solusi[j].z = new_solusi[j].z, new_solusi[i].z

        elif mutation_type == 'move':
            i = random.randint(0, len(new_solusi) - 1)
            panjang = new_solusi[i].barang.panjang if not new_solusi[i].rotated else new_solusi[i].barang.lebar
            lebar = new_solusi[i].barang.lebar if not new_solusi[i].rotated else new_solusi[i].barang.panjang

            max_x = max(0, self.mobil.panjang_box - panjang)
            max_y = max(0, self.mobil.lebar_box - lebar)
            max_z = max(0, self.mobil.tinggi_box - new_solusi[i].barang.tinggi)

            if max_x > 0 and max_y > 0 and max_z > 0:
                new_solusi[i].x = random.uniform(0, max_x)
                new_solusi[i].y = random.uniform(0, max_y)
                new_solusi[i].z = random.uniform(0, max_z)

        elif mutation_type == 'rotate':
            i = random.randint(0, len(new_solusi) - 1)
            new_solusi[i].rotated = not new_solusi[i].rotated

            panjang = new_solusi[i].barang.panjang if not new_solusi[i].rotated else new_solusi[i].barang.lebar
            lebar = new_solusi[i].barang.lebar if not new_solusi[i].rotated else new_solusi[i].barang.panjang

            if not self._check_fit_with_dimensions(panjang, lebar, new_solusi[i].barang.tinggi,
                                                   new_solusi[i].x, new_solusi[i].y, new_solusi[i].z):
                new_solusi[i].rotated = not new_solusi[i].rotated

        return new_solusi

    def optimize(self) -> List[PosisiBarang]:
        """Main SA optimization loop"""
        current_solution = self.generate_initial_solution()
        current_fitness = self.evaluate_fitness(current_solution)
        best_solution = [PosisiBarang(pos.barang, pos.x, pos.y, pos.z, pos.rotated) for pos in current_solution]
        best_fitness = current_fitness

        temperature = self.suhu_awal

        for iteration in range(self.max_iterasi):
            neighbor_solution = self.mutate_solution(current_solution)
            neighbor_fitness = self.evaluate_fitness(neighbor_solution)

            delta = neighbor_fitness - current_fitness

            if delta > 0 or (temperature > 0 and random.random() < math.exp(delta / temperature)):
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness

                if neighbor_fitness > best_fitness:
                    best_solution = [PosisiBarang(pos.barang, pos.x, pos.y, pos.z, pos.rotated) for pos in
                                     neighbor_solution]
                    best_fitness = neighbor_fitness

            temperature *= self.cooling_rate
            if temperature < self.suhu_min:
                break

        return best_solution


# =============================================================================
# ROUTE OPTIMIZER
# =============================================================================

class RouteOptimizer:
    def __init__(self, jarak_matrix: Dict[Tuple[str, str], float]):
        self.jarak_matrix = jarak_matrix

    def nearest_neighbor(self, kota_list: List[str], kota_asal: str = "Jakarta") -> Tuple[List[str], float]:
        """Algoritma Nearest Neighbor untuk TSP"""
        if not kota_list:
            return [kota_asal], 0

        unique_kota = list(set(kota_list))
        if kota_asal not in unique_kota:
            unique_kota.insert(0, kota_asal)

        unvisited = set(unique_kota)
        current_city = kota_asal
        route = [current_city]
        unvisited.remove(current_city)
        total_distance = 0

        while unvisited:
            distances = []
            for city in unvisited:
                dist = self.jarak_matrix.get((current_city, city), float('inf'))
                if dist != float('inf'):
                    distances.append((city, dist))

            if not distances:
                nearest_city = list(unvisited)[0]
                distance = 100  # Default distance
            else:
                nearest_city, distance = min(distances, key=lambda x: x[1])

            total_distance += distance
            route.append(nearest_city)
            unvisited.remove(nearest_city)
            current_city = nearest_city

        return route, total_distance

    def optimize_route(self, kota_list: List[str], kota_asal: str = "Jakarta") -> Tuple[List[str], float]:
        """Main function untuk optimasi rute"""
        if not kota_list:
            return [kota_asal], 0
        return self.nearest_neighbor(kota_list, kota_asal)


# =============================================================================
# MAIN HYBRID SYSTEM - FIXED VERSION
# =============================================================================

class HybridDeliverySystem:
    def __init__(self):
        self.fuzzy_engine = FuzzyLogicEngine()
        self.jarak_matrix = self._init_jarak_matrix()
        self.route_optimizer = RouteOptimizer(self.jarak_matrix)

    def _init_jarak_matrix(self) -> Dict[Tuple[str, str], float]:
        """Initialize jarak antar kota"""
        return {
            ("Jakarta", "Bandung"): 150,
            ("Bandung", "Jakarta"): 150,
            ("Jakarta", "Surabaya"): 800,
            ("Surabaya", "Jakarta"): 800,
            ("Bandung", "Surabaya"): 700,
            ("Surabaya", "Bandung"): 700,
            ("Jakarta", "Yogyakarta"): 560,
            ("Yogyakarta", "Jakarta"): 560,
            ("Bandung", "Yogyakarta"): 450,
            ("Yogyakarta", "Bandung"): 450,
            ("Surabaya", "Yogyakarta"): 320,
            ("Yogyakarta", "Surabaya"): 320,
        }

    def alokasi_barang_ke_mobil(self, barang_list: List[Barang], mobil_list: List[Mobil]) -> Dict[str, List[Barang]]:
        """FIXED: Alokasi barang ke mobil dengan algoritma yang lebih baik"""
        if not barang_list or not mobil_list:
            return {}

        # Initialize alokasi
        alokasi = {mobil.id: [] for mobil in mobil_list}

        # Sort barang berdasarkan volume dan prioritas (barang besar dan prioritas tinggi dulu)
        barang_sorted = sorted(barang_list,
                               key=lambda x: (x.prioritas, x.volume, x.berat),
                               reverse=True)

        print(f"🔄 Mengalokasikan {len(barang_sorted)} barang ke {len(mobil_list)} mobil...")

        for barang in barang_sorted:
            allocated = False

            # Cari mobil terbaik berdasarkan kapasitas sisa
            mobil_candidates = []

            for mobil in mobil_list:
                current_weight = sum([b.berat for b in alokasi[mobil.id]])
                current_volume = sum([b.volume for b in alokasi[mobil.id]])

                # Cek apakah barang muat
                if (current_weight + barang.berat <= mobil.kapasitas_berat and
                        current_volume + barang.volume <= mobil.volume_box):
                    # Hitung efisiensi penggunaan setelah menambah barang
                    weight_efficiency = (current_weight + barang.berat) / mobil.kapasitas_berat
                    volume_efficiency = (current_volume + barang.volume) / mobil.volume_box
                    combined_efficiency = (weight_efficiency + volume_efficiency) / 2

                    mobil_candidates.append((mobil, combined_efficiency))

            # Pilih mobil dengan efisiensi terbaik (paling seimbang)
            if mobil_candidates:
                # Sort berdasarkan efisiensi, pilih yang paling optimal
                mobil_candidates.sort(key=lambda x: x[1])
                best_mobil = mobil_candidates[0][0]
                alokasi[best_mobil.id].append(barang)
                allocated = True
                print(f"   ✅ {barang.nama} → {best_mobil.plat_nomor}")

            if not allocated:
                print(f"   ❌ {barang.nama} tidak dapat dialokasikan ke mobil manapun")

        # Print summary
        print(f"\n📊 Ringkasan Alokasi:")
        for mobil in mobil_list:
            jumlah_barang = len(alokasi[mobil.id])
            total_berat = sum([b.berat for b in alokasi[mobil.id]])
            total_volume = sum([b.volume for b in alokasi[mobil.id]])

            weight_usage = (total_berat / mobil.kapasitas_berat) * 100
            volume_usage = (total_volume / mobil.volume_box) * 100

            print(f"   🚛 {mobil.plat_nomor}: {jumlah_barang} barang, "
                  f"Berat: {weight_usage:.1f}%, Volume: {volume_usage:.1f}%")

        return alokasi

    def optimize_delivery(self, barang_list: List[Barang], mobil_list: List[Mobil]) -> Dict:
        """Main optimization function - FIXED"""
        if not barang_list or not mobil_list:
            return {}

        print(f"🚀 Memulai optimasi pengiriman untuk {len(barang_list)} barang dan {len(mobil_list)} mobil")

        # Step 1: Alokasi barang ke mobil
        alokasi = self.alokasi_barang_ke_mobil(barang_list, mobil_list)

        results = {}

        for mobil in mobil_list:
            barang_mobil = alokasi.get(mobil.id, [])

            if not barang_mobil:
                print(f"⏭️  Mobil {mobil.plat_nomor} tidak mendapat alokasi barang")
                continue

            print(f"\n🚛 Mengoptimasi {mobil.plat_nomor} dengan {len(barang_mobil)} barang...")

            # Step 2: SA untuk penempatan barang
            sa_optimizer = SimulatedAnnealing(mobil, barang_mobil, self.fuzzy_engine)
            penempatan_optimal = sa_optimizer.optimize()

            # Step 3: Optimasi rute kota
            kota_tujuan = [barang.kota_tujuan for barang in barang_mobil]
            rute_optimal, total_jarak = self.route_optimizer.optimize_route(kota_tujuan)

            # Step 4: Hitung biaya dan efisiensi
            total_biaya = total_jarak * mobil.biaya_per_km
            volume_used = sum([b.volume for b in barang_mobil])
            weight_used = sum([b.berat for b in barang_mobil])

            efisiensi_volume = (volume_used / mobil.volume_box * 100) if mobil.volume_box > 0 else 0
            efisiensi_berat = (weight_used / mobil.kapasitas_berat * 100) if mobil.kapasitas_berat > 0 else 0

            results[mobil.id] = {
                'mobil': mobil,
                'barang': barang_mobil,
                'penempatan': penempatan_optimal,
                'rute': rute_optimal,
                'total_jarak': total_jarak,
                'total_biaya': total_biaya,
                'efisiensi_volume': efisiensi_volume,
                'efisiensi_berat': efisiensi_berat,
                'jumlah_barang_ditempatkan': len(penempatan_optimal)
            }

        return results

    def print_results(self, results: Dict):
        """ENHANCED: Print hasil optimasi dengan informasi lebih detail"""
        if not results:
            print("❌ Tidak ada hasil optimasi untuk ditampilkan.")
            return

        print("\n" + "=" * 90)
        print("🎯 HASIL OPTIMASI PENGIRIMAN BARANG HYBRID SYSTEM")
        print("=" * 90)

        total_biaya_semua = 0
        total_barang_semua = 0
        total_jarak_semua = 0

        for mobil_id, result in results.items():
            mobil = result['mobil']
            barang_list = result['barang']
            penempatan = result['penempatan']

            print(f"\n🚛 MOBIL: {mobil.plat_nomor} (ID: {mobil.id})")
            print(f"   📦 Barang Dialokasikan: {len(barang_list)}")
            print(f"   📍 Barang Ditempatkan: {result['jumlah_barang_ditempatkan']}")
            print(f"   🛣️  Rute: {' → '.join(result['rute'])}")
            print(f"   📏 Total Jarak: {result['total_jarak']:.1f} km")
            print(f"   💰 Total Biaya: Rp {result['total_biaya']:,.0f}")
            print(f"   📊 Efisiensi Volume: {result['efisiensi_volume']:.1f}%")
            print(f"   ⚖️  Efisiensi Berat: {result['efisiensi_berat']:.1f}%")

            # Detail penempatan barang
            print(f"   📋 Detail Penempatan Barang:")
            penempatan_dict = {pos.barang.id: pos for pos in penempatan}

            for i, barang in enumerate(barang_list):
                if barang.id in penempatan_dict:
                    pos = penempatan_dict[barang.id]
                    rotation_info = " (Rotated)" if pos.rotated else ""
                    rapuh_info = " 🔴" if barang.rapuh else ""
                    prioritas_info = " ⭐" if barang.prioritas >= 2 else ""
                    print(f"      • {barang.nama}{rapuh_info}{prioritas_info} → {barang.kota_tujuan} | "
                          f"Pos: ({pos.x:.0f}, {pos.y:.0f}, {pos.z:.0f}){rotation_info}")
                else:
                    print(f"      • {barang.nama} → {barang.kota_tujuan} | ❌ Tidak ditempatkan")

            # Statistik per kota
            kota_stats = {}
            for barang in barang_list:
                kota = barang.kota_tujuan
                if kota not in kota_stats:
                    kota_stats[kota] = {'count': 0, 'weight': 0, 'volume': 0}
                kota_stats[kota]['count'] += 1
                kota_stats[kota]['weight'] += barang.berat
                kota_stats[kota]['volume'] += barang.volume

            print(f"   🏙️  Statistik per Kota:")
            for kota, stats in kota_stats.items():
                print(f"      • {kota}: {stats['count']} barang, "
                      f"{stats['weight']:.1f} kg, {stats['volume']:,.0f} cm³")

            total_biaya_semua += result['total_biaya']
            total_barang_semua += len(barang_list)
            total_jarak_semua += result['total_jarak']

        print(f"\n" + "=" * 90)
        print(f"💰 TOTAL BIAYA SEMUA MOBIL: Rp {total_biaya_semua:,.0f}")
        print(f"📦 TOTAL BARANG DIANGKUT: {total_barang_semua}")
        print(f"📏 TOTAL JARAK TEMPUH: {total_jarak_semua:.1f} km")
        print(f"🚛 JUMLAH MOBIL DIGUNAKAN: {len(results)}")
        print("=" * 90)


# =============================================================================
# DEMO / TESTING - ENHANCED
# =============================================================================

import random
import numpy as np

# Set seed untuk hasil yang konsisten
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def demo_system():
    """Demo sistem hybrid"""
    # Data sample
    set_random_seed(42)  # Gunakan angka seed yang sama

    barang_list = [
        Barang("B001", "Laptop Dell", 2.5, 35, 25, 3, "Bandung", rapuh=True, prioritas=2),
        Barang("B002", "Rice Cooker", 8.0, 40, 30, 25, "Surabaya"),
        Barang("B003", "Buku Paket", 15.0, 30, 20, 40, "Yogyakarta"),
        Barang("B004", "Kulkas Mini", 25.0, 60, 50, 80, "Bandung"),
        Barang("B005", "Smartphone", 0.5, 15, 8, 1, "Jakarta", rapuh=True, prioritas=2),
        Barang("B006", "Meja Kayu", 30.0, 120, 60, 5, "Surabaya"),
        Barang("B007", "Printer Canon", 6.5, 45, 35, 20, "Yogyakarta"),
        Barang("B008", "Kipas Angin", 5.0, 40, 30, 70, "Jakarta"),
        Barang("B009", "Paket Makanan", 10.0, 30, 30, 30, "Bandung", rapuh=True, prioritas=1),
        Barang("B010", "Monitor LED", 7.5, 50, 40, 15, "Surabaya", rapuh=True),
        Barang("B011", "Koper Baju", 12.0, 60, 45, 25, "Yogyakarta"),
        Barang("B012", "TV 32 inch", 20.0, 80, 50, 10, "Jakarta"),
        Barang("B013", "Dispenser", 9.0, 35, 35, 60, "Bandung"),
        Barang("B014", "Speaker Aktif", 11.0, 40, 40, 50, "Surabaya", rapuh=True),
        Barang("B015", "Setrika", 3.0, 25, 15, 10, "Yogyakarta"),
        Barang("B016", "Camera DSLR", 1.2, 20, 15, 10, "Jakarta", rapuh=True, prioritas=2),
        Barang("B017", "Karpet", 13.0, 100, 20, 20, "Bandung"),
        Barang("B018", "Vacuum Cleaner", 8.5, 50, 35, 35, "Surabaya"),
        Barang("B019", "Mainan Anak", 4.0, 40, 30, 20, "Yogyakarta"),
        Barang("B020", "Rak Buku", 22.0, 90, 40, 10, "Jakarta"),
        Barang("B021", "PC Gaming", 18.0, 60, 60, 60, "Bandung", rapuh=True, prioritas=1),
        Barang("B022", "Sepatu Kulit", 2.0, 35, 20, 12, "Surabaya"),
        Barang("B023", "Kamera CCTV", 1.5, 15, 15, 15, "Yogyakarta", rapuh=True),
        Barang("B024", "Toaster", 2.8, 25, 20, 20, "Jakarta"),
        Barang("B025", "Alat Fitness", 35.0, 150, 60, 60, "Bandung"),
        Barang("B026", "Microwave", 14.0, 60, 50, 40, "Surabaya"),
        Barang("B027", "Jam Dinding", 1.0, 30, 30, 5, "Yogyakarta"),
        Barang("B028", "Set Alat Masak", 6.0, 40, 35, 20, "Jakarta"),
        Barang("B029", "Tas Sekolah", 3.0, 35, 25, 20, "Bandung"),
        Barang("B030", "Bantal Guling", 4.0, 80, 30, 30, "Surabaya"),
        Barang("B031", "Kompor Gas", 9.0, 60, 45, 35, "Bandung"),
        Barang("B032", "Keyboard Mechanical", 1.2, 45, 15, 5, "Surabaya", rapuh=True),
        Barang("B033", "Boneka Panda", 2.8, 40, 30, 25, "Yogyakarta"),
        Barang("B034", "Box Arsip", 5.5, 60, 40, 30, "Jakarta"),
        Barang("B035", "Sofa Kecil", 32.0, 120, 70, 80, "Bandung"),
        Barang("B036", "Meja Belajar", 25.0, 110, 50, 75, "Surabaya"),
        Barang("B037", "Rak TV", 22.5, 100, 40, 60, "Yogyakarta"),
        Barang("B038", "Blender", 3.5, 30, 25, 35, "Jakarta", rapuh=True),
        Barang("B039", "Kursi Lipat", 6.0, 90, 45, 10, "Bandung"),
        Barang("B040", "Jam Alarm", 1.0, 15, 10, 10, "Surabaya", rapuh=True),
        Barang("B041", "Matras Yoga", 4.0, 100, 20, 20, "Yogyakarta"),
        Barang("B042", "Lukisan Kanvas", 2.0, 70, 50, 5, "Jakarta", rapuh=True),
        Barang("B043", "Router WiFi", 0.8, 20, 20, 10, "Bandung"),
        Barang("B044", "Kamera Analog", 1.3, 20, 15, 12, "Surabaya", rapuh=True, prioritas=2),
        Barang("B045", "Tenda Camping", 10.0, 80, 40, 30, "Yogyakarta"),
        Barang("B046", "Bingkai Foto", 1.5, 25, 20, 3, "Jakarta", rapuh=True),
        Barang("B047", "Power Bank", 0.6, 10, 8, 4, "Bandung", rapuh=True),
        Barang("B048", "Projector", 4.0, 40, 30, 15, "Surabaya", rapuh=True),
        Barang("B049", "Drone Mini", 1.0, 20, 20, 10, "Yogyakarta", rapuh=True, prioritas=2),
        Barang("B050", "Skateboard", 6.0, 80, 25, 15, "Jakarta"),
        Barang("B051", "Mesin Jahit", 14.0, 60, 45, 35, "Bandung"),
        Barang("B052", "Box Peralatan", 12.0, 50, 40, 35, "Surabaya"),
        Barang("B053", "Kulkas 2 Pintu", 45.0, 140, 70, 70, "Yogyakarta"),
        Barang("B054", "TV 42 inch", 25.0, 95, 60, 10, "Jakarta", rapuh=True),
        Barang("B055", "Matras Lantai", 6.5, 100, 40, 20, "Bandung"),
        Barang("B056", "Perlengkapan Bayi", 3.5, 40, 35, 25, "Surabaya"),
        Barang("B057", "Speaker Bluetooth", 1.8, 20, 20, 15, "Yogyakarta"),
        Barang("B058", "Kotak Hadiah", 2.2, 30, 30, 15, "Jakarta"),
        Barang("B059", "Game Console", 2.3, 30, 25, 10, "Bandung", rapuh=True),
        Barang("B060", "Tas Gunung", 7.0, 70, 40, 30, "Surabaya"),
        Barang("B061", "Lemari Plastik", 18.0, 100, 60, 50, "Yogyakarta"),
        Barang("B062", "Alat Panggang", 11.0, 70, 50, 35, "Jakarta"),
        Barang("B063", "Kamera Mirrorless", 1.5, 20, 15, 10, "Bandung", rapuh=True, prioritas=2),
        Barang("B064", "Vacuum Portable", 3.0, 35, 30, 20, "Surabaya"),
        Barang("B065", "Kursi Gaming", 28.0, 120, 60, 70, "Yogyakarta"),
        Barang("B066", "Rak Sepatu", 9.0, 80, 30, 90, "Jakarta"),
        Barang("B067", "Jam Kayu", 2.5, 40, 40, 8, "Bandung", rapuh=True),
        Barang("B068", "Pisau Set", 2.0, 35, 20, 10, "Surabaya", rapuh=True),
        Barang("B069", "Mesin Kopi", 7.0, 50, 35, 40, "Yogyakarta"),
        Barang("B070", "Handuk Besar", 2.8, 60, 30, 15, "Jakarta"),
        Barang("B071", "Kamera Web", 0.7, 12, 8, 6, "Bandung", rapuh=True),
        Barang("B072", "TV 50 inch", 30.0, 110, 70, 12, "Surabaya", rapuh=True),
        Barang("B073", "Speaker Subwoofer", 14.5, 60, 50, 50, "Yogyakarta"),
        Barang("B074", "Koper Besar", 16.0, 80, 60, 35, "Jakarta"),
        Barang("B075", "Modem Internet", 0.6, 15, 10, 5, "Bandung", rapuh=True),
        Barang("B076", "Frame Foto", 1.1, 30, 25, 2, "Surabaya"),
        Barang("B077", "Gitar Akustik", 6.0, 100, 40, 12, "Yogyakarta", rapuh=True),
        Barang("B078", "Lemari Kayu", 40.0, 150, 70, 80, "Jakarta"),
        Barang("B079", "Bantal Sofa", 3.5, 40, 40, 15, "Bandung"),
        Barang("B080", "Camera GoPro", 1.0, 15, 10, 10, "Surabaya", rapuh=True),
        Barang("B081", "Mesin Pemotong", 20.0, 80, 60, 50, "Yogyakarta"),
        Barang("B082", "Raket Badminton", 1.5, 70, 25, 8, "Jakarta"),
        Barang("B083", "Peralatan Bengkel", 10.0, 60, 50, 35, "Bandung"),
        Barang("B084", "Jam Tangan", 0.5, 10, 8, 4, "Surabaya", rapuh=True, prioritas=2),
        Barang("B085", "TV 55 inch", 35.0, 120, 75, 10, "Yogyakarta", rapuh=True),
        Barang("B086", "AC Portable", 29.0, 80, 60, 70, "Jakarta"),
        Barang("B087", "Gitar Listrik", 7.0, 100, 40, 15, "Bandung", rapuh=True),
        Barang("B088", "Drone Kamera", 1.3, 25, 25, 12, "Surabaya", rapuh=True),
        Barang("B089", "Set Panci", 4.5, 40, 35, 25, "Yogyakarta"),
        Barang("B090", "Lemari Es", 48.0, 160, 80, 70, "Jakarta"),
        Barang("B091", "Helm Fullface", 2.8, 35, 30, 25, "Bandung"),
        Barang("B092", "Jam Digital", 1.2, 25, 15, 10, "Surabaya", rapuh=True),
        Barang("B093", "Soundbar", 3.0, 80, 10, 15, "Yogyakarta", rapuh=True),
        Barang("B094", "Hoodie", 1.0, 40, 30, 10, "Jakarta"),
        Barang("B095", "Laptop Asus", 2.2, 35, 25, 3, "Bandung", rapuh=True, prioritas=2),
        Barang("B096", "Panci Listrik", 5.0, 40, 35, 30, "Surabaya"),
        Barang("B097", "Timbangan Digital", 1.4, 25, 20, 10, "Yogyakarta"),
        Barang("B098", "AC Split", 33.0, 130, 70, 30, "Jakarta"),
        Barang("B099", "Meja Rias", 26.0, 100, 50, 80, "Bandung"),
        Barang("B100", "Peralatan Melukis", 3.2, 40, 30, 20, "Surabaya", rapuh=True)
    ]
    # FIX: Remove duplicate mobil entries
    mobil_list = [
        Mobil("M001", "B 1234 XYZ", 1000, 300, 180, 180, 5000),
        Mobil("M002", "B 5678 ABC", 800, 250, 150, 150, 4500),
        Mobil("M003", "B 9999 DEF", 1200, 350, 200, 200, 5500),
    ]

    # Jalankan sistem
    system = HybridDeliverySystem()

    try:
        results = system.optimize_delivery(barang_list, mobil_list)
        system.print_results(results)
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        import traceback
        traceback.print_exc()


class DeterministicHybridSystem(HybridDeliverySystem):
    def __init__(self, seed=42):
        super().__init__()
        self.seed = seed

    def optimize_delivery(self, barang_list, mobil_list):
        # Set seed setiap kali optimasi dimulai
        random.seed(self.seed)
        np.random.seed(self.seed)

        return super().optimize_delivery(barang_list, mobil_list)


if __name__ == "__main__":
    demo_system()
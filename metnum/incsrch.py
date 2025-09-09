import math

# Fungsi untuk hitung f(c)
def f(c):
    m = 68.1  # massa parachutist
    g = 9.8   # gravitasi
    t = 10    # waktu
    v = 40    # kecepatan yang diinginkan
    return (g * m / c) * (1 - math.exp(-c / m * t)) - v

# Parameter incremental search
c_awal = 1       # mulai dari c = 1
c_akhir = 15     # sampai c = 15
langkah_besar = 1  # langkah kasar
langkah_kecil = 0.1  # langkah halus
toleransi = 0.001  # kapan bilang "udah deket nol"

# Loop luar: cek rentang besar
c = c_awal
nilai_sebelumnya = f(c)

while c <= c_akhir:
    nilai_sekarang = f(c)
    # cek kalau f(c) berubah tanda
    if nilai_sekarang * nilai_sebelumnya < 0:
        print(f"Tanda berubah di antara {c - langkah_besar} dan {c}!")
        # Loop dalam: cek rentang kecil
        c_kecil = c - langkah_besar
        if (nilai_sekarang - nilai_sebelumnya)/nilai_sekarang >=0.05 and nilai_sekarang != nilai_sebelumnya:
            c_s = f(c_kecil)
            while c_kecil <= c:
                nilai_kecil = f(c_kecil)
                if nilai_kecil == c_s:
                    c_kecil += langkah_kecil
                    continue
                print(c_kecil, ". ", nilai_kecil)
                if nilai_kecil != 0 and (nilai_kecil - c_s) / nilai_kecil > 0.05:
                    print("tess")
                    if nilai_kecil * c_s < 0:
                        c_s = f(c_kecil-(langkah_kecil*2))
                        c_kecil -= langkah_kecil
                        langkah_kecil /= 10
                        continue

                    if langkah_kecil == toleransi:
                        print(c_kecil, ". ", nilai_kecil)
                        break
                    c_kecil += langkah_kecil
                    c_s = nilai_kecil
                else:
                    print((nilai_kecil-c_s)/nilai_kecil)
                    break
        else:
            break
    nilai_sebelumnya = nilai_sekarang
    c += langkah_besar

print("Selesai nyari!")
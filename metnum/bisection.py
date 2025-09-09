import math


# Fungsi untuk hitung f(c)
def f(c):
    m = 68.1  # massa parachutist
    g = 9.8  # gravitasi
    t = 10  # waktu
    v = 40  # kecepatan yang diinginkan
    return (g * m / c) * (1 - math.exp(-c / m * t)) - v


# Metode bisection
def bisection(xl, xu, toleransi=0.001, max_iterasi=100, x_true=14.78):
    iterasi = 0
    while iterasi < max_iterasi:
        xr = (xl + xu) / 2  # Titik tengah
        f_xl = f(xl)
        f_xu = f(xu)
        f_xr = f(xr)

        # Hitung approximate error
        if iterasi > 0:
            ea = abs((xr - xr_lama) / xr) * 100
        else:
            ea = 100  # Error awal besar

        # Hitung true error (berdasarkan x_true perkiraan)
        et = abs((x_true - xr) / x_true) * 100

        print(f"Iterasi {iterasi + 1}: xl = {xl}, xu = {xu}, xr = {xr}, ea = {ea:.3f}%, et = {et:.3f}%")

        # Cek konvergensi
        if ea < toleransi:
            print(f"Hampir ketemu! c = {xr}, f(c) = {f_xr}, error ea = {ea:.3f}%, et = {et:.3f}%")
            return xr

        # Tentuin batas baru
        if f_xl * f_xr < 0:
            xu = xr
        elif f_xr * f_xu < 0:
            xl = xr
        else:
            print("Tidak ada akar di interval ini!")
            return None

        xr_lama = xr
        iterasi += 1

    print("Mencapai batas iterasi!")
    return xr


# Jalankan bisection dengan interval awal dari contoh
xu_awal = 16
xl_awal = 12
hasil = bisection(xl_awal, xu_awal, x_true=14.7801)  # Pake perkiraan x_true
print(f"Akar perkiraan: {hasil}")
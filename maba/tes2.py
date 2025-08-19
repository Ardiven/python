import pandas as pd
import random

# Step 1: Load data lengkap
df = pd.read_excel("FKG.xlsx")

# Step 2: Filter berdasarkan beberapa Program Studi
input_prodi = input("Masukkan nama-nama prodi dipisah koma (misal: KEDOKTERAN, FARMASI): ")
daftar_prodi = [p.strip().upper() for p in input_prodi.split(",")]
df_filtered = df[df['Program Studi'].str.upper().isin(daftar_prodi)].copy()

# Step 3: Pisahkan berdasarkan agama Kristen & Katolik vs lainnya
agama_kk = ['Kristen', 'Katolik']
df_kk = df_filtered[df_filtered['Religion'].isin(agama_kk)].copy()
df_lain = df_filtered[~df_filtered['Religion'].isin(agama_kk)].copy()

# Fungsi pisah gender dan acak
def pisah_gender(df_):
    cowok = df_[df_['Gender'] == 'Laki-laki'].to_dict('records')
    cewek = df_[df_['Gender'] == 'Perempuan'].to_dict('records')
    random.shuffle(cowok)
    random.shuffle(cewek)
    return cowok, cewek

kk_cowok, kk_cewek = pisah_gender(df_kk)
lain_cowok, lain_cewek = pisah_gender(df_lain)

# Gabungkan cowok dan cewek dari KK dan lainnya
cowok_list = kk_cowok + lain_cowok
cewek_list = kk_cewek + lain_cewek
random.shuffle(cowok_list)
random.shuffle(cewek_list)

# Step 4: Tentukan jumlah kelompok
jumlah_kelompok = int(input("Jumlah kelompok: "))
groups = [[] for _ in range(jumlah_kelompok)]

# Fungsi cek sekolah unik
def bisa_masuk(group, maba):
    sekolahs = [anggota['Asal Sekolah'] for anggota in group]
    return maba['Asal Sekolah'] not in sekolahs

# Fungsi cek agama maksimal 3 unik per kelompok
def agama_belum_penuh(group, maba):
    agama_group = set(anggota['Religion'] for anggota in group)
    return maba['Religion'] in agama_group or len(agama_group) < 3

# Step 5: Isi minimal 2 cowok dan 2 cewek per kelompok
def ambil_kandidat(kandidat_list, group, jumlah):
    count = 0
    idx = 0
    while count < jumlah and idx < len(kandidat_list):
        kandidat = kandidat_list[idx]
        if bisa_masuk(group, kandidat) and agama_belum_penuh(group, kandidat):
            group.append(kandidat)
            kandidat_list.pop(idx)
            count += 1
        else:
            idx += 1
    return count

for i in range(jumlah_kelompok):
    ambil_kandidat(cowok_list, groups[i], 2)
    ambil_kandidat(cewek_list, groups[i], 2)

# Step 6: Distribusikan sisa mahasiswa ke kelompok dengan aturan unik sekolah & max 3 agama
sisa_maba = cowok_list + cewek_list
random.shuffle(sisa_maba)

for maba in sisa_maba:
    masuk = False
    sorted_groups = sorted(groups, key=lambda g: len(g))
    for group in sorted_groups:
        if bisa_masuk(group, maba) and agama_belum_penuh(group, maba):
            group.append(maba)
            masuk = True
            break
    if not masuk:
        min_group = min(groups, key=lambda g: len(g))
        min_group.append(maba)

# Step 7: Buat DataFrame hasil dengan kolom Kelompok
hasil_list = []
for i, group in enumerate(groups, 1):
    for anggota in group:
        anggota_copy = anggota.copy()
        anggota_copy['Kelompok'] = i
        hasil_list.append(anggota_copy)

df_hasil = pd.DataFrame(hasil_list)

# Step 8: Simpan ke CSV baru
output_file = "GABUNGAN_" + "_".join(p.replace(" ", "_") for p in daftar_prodi) + ".csv"
df_hasil.to_csv(output_file, index=False)
print(f"Hasil pembagian kelompok berhasil disimpan ke '{output_file}'")

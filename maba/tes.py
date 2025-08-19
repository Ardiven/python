import pandas as pd
import random

# Load data lengkap
df = pd.read_excel("FKG.xlsx")

# Filter berdasarkan Program Studi
prodi_yang_diinginkan = "KEDOKTERAN GIGI"
df_filtered = df[df['Program Studi'] == prodi_yang_diinginkan].copy()

# Pisahkan berdasarkan agama Kristen & Katolik vs lain
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

groups = []

# Fungsi cek sekolah unik
def bisa_masuk(group, maba):
    sekolahs = [anggota['Asal Sekolah'] for anggota in group]
    return maba['Asal Sekolah'] not in sekolahs

# Fungsi cek agama maksimal 3 unik per kelompok
def agama_belum_penuh(group, maba):
    agama_group = set(anggota['Religion'] for anggota in group)
    if maba['Religion'] in agama_group:
        return True
    else:
        return len(agama_group) < 3

# Step 1: Buat sebanyak mungkin kelompok hanya Kristen & Katolik dengan minimal 2 cowok + 2 cewek
while len(kk_cowok) >= 2 and len(kk_cewek) >= 2:
    group_baru = []
    # Ambil 2 cowok
    count = 0
    idx = 0
    while count < 2 and idx < len(kk_cowok):
        kandidat = kk_cowok[idx]
        if bisa_masuk(group_baru, kandidat):
            group_baru.append(kandidat)
            kk_cowok.pop(idx)
            count += 1
        else:
            idx += 1
    # Jika kurang dari 2 cowok, tapi masih ada, paksakan ambil tanpa cek sekolah unik (karena harus minimal 2)
    while count < 2 and len(kk_cowok) > 0:
        kandidat = kk_cowok.pop(0)
        group_baru.append(kandidat)
        count += 1

    # Ambil 2 cewek dengan cara sama
    count = 0
    idx = 0
    while count < 2 and idx < len(kk_cewek):
        kandidat = kk_cewek[idx]
        if bisa_masuk(group_baru, kandidat):
            group_baru.append(kandidat)
            kk_cewek.pop(idx)
            count += 1
        else:
            idx += 1
    while count < 2 and len(kk_cewek) > 0:
        kandidat = kk_cewek.pop(0)
        group_baru.append(kandidat)
        count += 1

    groups.append(group_baru)

# Step 2: Gabungkan sisa semua maba yang belum terbagi (baik KK maupun lain)
cowok_list = kk_cowok + lain_cowok
cewek_list = kk_cewek + lain_cewek
random.shuffle(cowok_list)
random.shuffle(cewek_list)

# Tentukan jumlah kelompok untuk sisa (misal jumlah kelompok yang sudah terbentuk + 4 baru)
jumlah_kelompok_sisa = 4
groups_sisa = [[] for _ in range(jumlah_kelompok_sisa)]

# Fungsi ambil kandidat untuk isi minimal 2 cowok dan 2 cewek per kelompok (cek agama max 3 dan sekolah unik)
def ambil_kandidat(kandidat_list, group, jumlah):
    count = 0
    idx = 0
    while count < jumlah and idx < len(kandidat_list):
        kandidat = kandidat_list[idx]
        # Cek sekolah unik dan agama max 3
        agama_group = set(anggota['Religion'] for anggota in group)
        if bisa_masuk(group, kandidat) and (kandidat['Religion'] in agama_group or len(agama_group) < 3):
            group.append(kandidat)
            kandidat_list.pop(idx)
            count += 1
        else:
            idx += 1
    return count

# Isi minimal 2 cowok dan 2 cewek di setiap kelompok sisa
for i in range(jumlah_kelompok_sisa):
    ambil_kandidat(cowok_list, groups_sisa[i], 2)
    ambil_kandidat(cewek_list, groups_sisa[i], 2)

# Step 3: Gabungkan sisa maba ke kelompok sisa dengan aturan sama
sisa_maba = cowok_list + cewek_list
random.shuffle(sisa_maba)

for maba in sisa_maba:
    masuk = False
    sorted_groups = sorted(groups_sisa, key=lambda g: len(g))
    for group in sorted_groups:
        agama_group = set(anggota['Religion'] for anggota in group)
        if bisa_masuk(group, maba) and (maba['Religion'] in agama_group or len(agama_group) < 3):
            group.append(maba)
            masuk = True
            break
    if not masuk:
        min_group = min(groups_sisa, key=lambda g: len(g))
        min_group.append(maba)

# Gabungkan kelompok KK yang sudah dibuat dengan kelompok sisa
groups.extend(groups_sisa)

# Step 4: Buat DataFrame hasil dengan kolom Kelompok
hasil_list = []
for i, group in enumerate(groups, 1):
    for anggota in group:
        anggota_copy = anggota.copy()
        anggota_copy['Kelompok'] = i
        hasil_list.append(anggota_copy)

df_hasil = pd.DataFrame(hasil_list)

# Step 5: Simpan ke CSV baru
output_file = "hasil_pembagian_kelompok_agama_only_kk_dan_campuran.csv"
df_hasil.to_csv(output_file, index=False)
print(f"Hasil pembagian kelompok berhasil disimpan ke '{output_file}'")

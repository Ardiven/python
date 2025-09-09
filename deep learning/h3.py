import numpy as np

# 1. Definisikan Source Layer
source_layer = np.array([
    [5, 2, 6, 8, 2, 0, 1, 2],
    [4, 3, 4, 5, 1, 9, 6, 3],
    [3, 9, 2, 4, 7, 7, 6, 9],
    [1, 3, 4, 6, 8, 2, 2, 1],
    [8, 4, 6, 2, 3, 1, 8, 8],
    [5, 8, 9, 0, 1, 0, 2, 3],
    [9, 2, 6, 6, 3, 6, 2, 1],
    [9, 8, 8, 2, 6, 3, 4, 5]
])

# 2. Definisikan Convolutional Kernel
kernel = np.array([
    [-1, 0, 1],
    [ 2, 1, 2],
    [ 1,-2, 0]
])

# Hitung padding yang dibutuhkan
pad_size = kernel.shape[0] // 2

# Tambahkan padding ke source_layer
padded_source_layer = np.pad(source_layer, pad_width=pad_size, mode='constant', constant_values=0)

# 3. Hitung dimensi output (sekarang sama dengan dimensi source_layer)
output_dim = source_layer.shape[0]
destination_layer = np.zeros((output_dim, output_dim))

# 4. Lakukan operasi konvolusi pada padded_source_layer
for i in range(output_dim):
    for j in range(output_dim):
        window = padded_source_layer[i:i+kernel.shape[0], j:j+kernel.shape[1]]
        convolution_result = np.sum(window * kernel)
        destination_layer[i, j] = convolution_result

# Mengubah tipe data menjadi integer
destination_layer = destination_layer.astype(int)

# 5. Simpan hasil ke file CSV
nama_file = 'hasil_konvolusi_padded.csv'
np.savetxt(nama_file, destination_layer, delimiter=',', fmt='%d')

# 6. Tampilkan hasil dan konfirmasi penyimpanan
print(f"Hasil Akhir (Destination Layer):\n{destination_layer}")
print(f"\n✅ Hasil telah berhasil disimpan ke dalam file '{nama_file}'")
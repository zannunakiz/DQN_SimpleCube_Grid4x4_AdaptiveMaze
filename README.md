# SKRPSI_CUBE

🎯 **SKRPSI_CUBE** adalah proyek edukatif berbasis Python yang menampilkan simulasi **Deep Q-Network (DQN)** pada lingkungan **GridWorld 4x4**. Repository ini juga dilengkapi dengan tool ilustrasi visual untuk membantu mendesain dan memahami representasi grid secara manual.

## ✨ Gambaran Umum

Proyek ini berfokus pada dua alur utama:

- **`main.py`** untuk menjalankan training agent DQN pada GridWorld.
- **`illustrate.py`** untuk membuka editor grid 4x4 interaktif berbasis Pygame.

Pendekatan yang digunakan sengaja dibuat ringkas dan mudah dibaca, sehingga cocok untuk:

- demonstrasi konsep reinforcement learning,
- eksperimen awal untuk skripsi atau pembelajaran,
- visualisasi perilaku agent pada lingkungan kecil yang terkontrol.

## 🧠 Fitur Utama

- Start dan goal berada di posisi tetap.
- Hole diacak pada setiap episode, tetapi tetap divalidasi agar masih tersedia jalur menuju goal.
- State agent hanya memakai 4 neuron lokal: `atas`, `bawah`, `kiri`, `kanan`.
- Agent menggunakan arsitektur DQN fully connected yang sederhana.
- Training dapat ditampilkan secara real-time melalui GUI Pygame.
- Editor grid manual tersedia untuk membantu ilustrasi skenario warna pada board.

## 🗂️ Struktur Codebase

- **`main.py`**  
  Menyediakan implementasi lingkungan GridWorld, replay buffer, network DQN, agent, proses training, dan CLI argument parser.

- **`illustrate.py`**  
  Tool visual untuk mewarnai grid 4x4 secara manual menggunakan mouse dan shortcut keyboard.

- **`requirements.txt`**  
  Daftar dependency utama yang dibutuhkan untuk menjalankan proyek.

## ⚙️ Kebutuhan Sistem

- Python 3.10 atau lebih baru direkomendasikan
- `pip` untuk instalasi package
- Lingkungan desktop dengan dukungan tampilan grafis untuk Pygame

## 📦 Instalasi

```bash
# Clone repository
git clone <url-repository>
cd SKRPSI_CUBE

# (Opsional) buat virtual environment
python -m venv .venv

# Aktifkan virtual environment di Windows PowerShell
.venv\Scripts\Activate.ps1

# Install semua dependency utama
pip install -r requirements.txt
```

## ▶️ Menjalankan Simulasi DQN

Gunakan file `main.py` untuk training agent dan visualisasi hasilnya.

```bash
# Jalankan training dengan pengaturan default dan GUI aktif
python main.py

# Jalankan training selama 300 episode
python main.py --episodes 300

# Batasi maksimal 25 langkah per episode
python main.py --episodes 300 --max-steps 25

# Render GUI setiap 3 langkah agar tampilan lebih ringan
python main.py --render-every 3 --fps 10

# Jalankan training tanpa membuka GUI
python main.py --no-render

# Tampilkan seluruh opsi CLI yang tersedia
python main.py --help
```

### Opsi CLI `main.py`

- `--episodes` : jumlah episode training yang akan dijalankan.
- `--max-steps` : batas maksimal langkah dalam satu episode.
- `--render-every` : interval render GUI per sejumlah step.
- `--fps` : batas frame rate untuk animasi Pygame.
- `--seed` : seed random agar eksperimen dapat direproduksi.
- `--no-render` : menonaktifkan GUI dan menjalankan training di terminal saja.

## 🎨 Menjalankan Grid Illustrator

Gunakan file `illustrate.py` untuk membuat ilustrasi warna grid secara manual.

```bash
# Buka editor grid 4x4 interaktif
python illustrate.py
```

### Kontrol di `illustrate.py`

- Klik tombol warna di panel kanan, lalu klik sel pada grid.
- Tekan `1` untuk putih.
- Tekan `2` untuk kuning.
- Tekan `3` untuk biru.
- Tekan `4` untuk merah.
- Tekan `5` untuk hijau.
- Tekan `C` untuk membersihkan seluruh grid.
- Tekan `S` untuk mencetak matriks warna ke terminal.
- Tekan `Esc` untuk keluar dari aplikasi.

## 🔬 Representasi State dan Reward

Pada `main.py`, state agent direpresentasikan sebagai:

```text
[atas, bawah, kiri, kanan]
```

Makna nilai setiap sensor:

- `1` = sel putih / clear
- `2` = sel kuning / pernah dikunjungi
- `3` = sel merah / hole
- `4` = wall / di luar batas grid
- `5` = sel hijau / goal

Mapping aksi agent:

- `0` = kiri
- `1` = kanan
- `2` = atas
- `3` = bawah

Reward default:

- `clear` = `+1`
- `yellow` = `-3`
- `wall` = `-10`
- `hole` = `-5`
- `goal` = `+20`

## 🔄 Alur Training Singkat

1. Environment di-reset dan hole acak dihasilkan.
2. Agent memilih aksi menggunakan strategi epsilon-greedy.
3. Transisi disimpan ke replay buffer.
4. Online network belajar dari batch pengalaman.
5. Target network diperbarui secara berkala.
6. Reward episode dicetak ke terminal sebagai log training.

## 📝 Catatan Penggunaan

- Proyek ini dirancang terutama untuk kebutuhan edukasi dan eksperimen kecil.
- Jika jendela Pygame ditutup saat training berjalan, proses akan dihentikan dengan aman.
- Karena grid berukuran kecil, fokus utama proyek ini adalah keterbacaan logika, bukan kompleksitas lingkungan.

## 👏 Credits

**Richky Abendego**  
Creator and Main Developer

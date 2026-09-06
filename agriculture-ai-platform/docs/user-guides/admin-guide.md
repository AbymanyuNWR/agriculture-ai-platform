# Admin Guide

## Overview

Panduan ini untuk administrator Agriculture AI Platform.

## Dashboard Admin

### Akses Dashboard

1. Buka https://admin.agriculture-ai.com
2. Login dengan akun admin
3. Masukkan email dan password
4. Klik "Login"

### Menu Dashboard

1. **Overview**: Statistik umum platform
2. **Users**: Manajemen pengguna
3. **Diagnosis**: Riwayat diagnosis
4. **Models**: Manajemen model ML
5. **System**: Pengaturan sistem
6. **Reports**: Laporan dan analitik

## Manajemen Pengguna

### Melihat Pengguna

1. Klik menu "Users"
2. Gunakan filter untuk pencarian
3. Klik nama pengguna untuk detail

### Menambah Pengguna

1. Klik "Add User"
2. Isi form:
   - Nama
   - Email
   - Nomor HP
   - Role (Admin/User)
3. Klik "Save"

### Mengedit Pengguna

1. Pilih pengguna
2. Klik "Edit"
3. Ubah informasi yang diperlukan
4. Klik "Save"

### Menghapus Pengguna

1. Pilih pengguna
2. Klik "Delete"
3. Konfirmasi penghapusan

## Manajemen Model

### Melihat Model

1. Klik menu "Models"
2. Lihat daftar model yang tersedia
3. Lihat versi dan performa

### Update Model

1. Pilih model
2. Klik "Update"
3. Upload file model baru
4. Konfigurasi parameter
5. Klik "Deploy"

### Monitor Performa

1. Pilih model
2. Lihat grafik performa
3. Cek metrik:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
4. Bandingkan dengan versi sebelumnya

## Manajemen Knowledge Graph

### Melihat Knowledge Base

1. Klik menu "Knowledge"
2. Jelajahi graf pengetahuan
3. Cari penyakit atau treatment

### Menambah Pengetahuan

1. Klik "Add Knowledge"
2. Isi form:
   - Nama penyakit
   - Gejala
   - Penyebab
   - Treatment
   - Pencegahan
3. Klik "Save"

### Mengedit Pengetahuan

1. Pilih entri
2. Klik "Edit"
3. Ubah informasi
4. Klik "Save"

## Monitoring Sistem

### Health Check

1. Klik menu "System"
2. Lihat status service:
   - API Server
   - ML Server
   - Database
   - Redis
   - Neo4j
3. Cek uptime dan response time

### Logs

1. Klik menu "Logs"
2. Filter berdasarkan:
   - Level (INFO, WARNING, ERROR)
   - Service
   - Waktu
3. Cari log tertentu

### Alerts

1. Klik menu "Alerts"
2. Lihat alert yang aktif
3. Konfigurasi notifikasi:
   - Email
   - Slack
   - WhatsApp

## Laporan

### Laporan Harian

1. Klik menu "Reports"
2. Pilih "Daily Report"
3. Lihat statistik:
   - Jumlah diagnosis
   - Penyakit paling umum
   - Akurasi model
4. Download laporan

### Laporan Mingguan

1. Pilih "Weekly Report"
2. Lihat tren:
   - Jumlah pengguna
   - Diagnosis per hari
   - Penyakit yang meningkat
3. Export ke PDF

### Laporan Kustom

1. Pilih "Custom Report"
2. Pilih rentang waktu
3. Pilih metrik yang ingin dilihat
4. Generate laporan

## Konfigurasi Sistem

### Pengaturan Umum

1. Klik menu "Settings"
2. Pengaturan umum:
   - Nama platform
   - Logo
   - Kontak
3. Klik "Save"

### Pengaturan Keamanan

1. Klik "Security"
2. Konfigurasi:
   - Password policy
   - Session timeout
   - API keys
3. Klik "Save"

### Pengaturan Email

1. Klik "Email"
2. Konfigurasi SMTP:
   - Server
   - Port
   - Username
   - Password
3. Test koneksi
4. Klik "Save"

## Backup & Recovery

### Backup Database

1. Klik menu "Backup"
2. Klik "Backup Now"
3. Pilih target backup
4. Klik "Start Backup"
5. Tunggu hingga selesai

### Restore Database

1. Pilih backup
2. Klik "Restore"
3. Konfirmasi restore
4. Tunggu hingga selesai

### Export Data

1. Klik "Export"
2. Pilih data yang ingin diexport
3. Pilih format (CSV, JSON, Excel)
4. Klik "Download"

## Troubleshooting

### Masalah: Tidak bisa login
- Reset password
- Cek email untuk link reset
- Hubungi support jika masalah berlanjut

### Masalah: Dashboard lambat
- Clear cache browser
- Cek koneksi internet
- Restart browser

### Masalah: Model tidak update
- Cek status ML server
- Cek logs untuk error
- Restart ML server jika perlu

## Kontak Support

- **Email**: admin@agriculture-ai.com
- **WhatsApp**: +62 812-3456-7890
- **Telepon**: 0800-1234-5678
- **Jam Operasional**: Senin-Jumat, 09.00-17.00 WIB

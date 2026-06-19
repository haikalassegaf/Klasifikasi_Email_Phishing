# app.py
import tkinter as tk
from tkinter import messagebox, filedialog
import os
import threading
import customtkinter as cctk

# Mengimpor logika pemrosesan data dari model_worker
from model_worker import run_experiment

# Pengaturan tema tampilan GUI
cctk.set_appearance_mode("System")
cctk.set_default_color_theme("blue")

# Koleksi dataset default awal
DATASETS = {
    "CEAS_08": "dataset/CEAS_08.csv",
    "Enron": "dataset/Enron.csv",
    "Phishing_Email": "dataset/Phishing_Email.csv",
    "SpamAssasin": "dataset/SpamAssasin.csv"
}

# Kombinasi metode ekstraksi fitur dan algoritma
METHODS = [
    "SVM + TF-IDF",
    "Naive Bayes + TF-IDF",
    "SVM + Word2Vec",
    "Gaussian NB + Word2Vec",
    "SVM + CountVectorizer",
    "Naive Bayes + CountVectorizer"
]

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Klasifikasi Email Multi-Metode")
        
        self.root.geometry("1100x700")
        
        # Mengatur Grid Layout Utama
        self.root.grid_columnconfigure(0, weight=1, minsize=360)
        self.root.grid_columnconfigure(1, weight=3)
        self.root.grid_rowconfigure(0, weight=1)

        # ==========================================
        # PANEL KIRI (KONTROL DAN OPSI)
        # ==========================================
        self.left_panel = cctk.CTkFrame(root, corner_radius=15)
        self.left_panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.title_label = cctk.CTkLabel(
            self.left_panel, 
            text="Email Classifier Dashboard", 
            font=cctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.pack(pady=(25, 20), padx=20)

        # Dropdown Menu - Pilih Dataset
        self.lbl_dataset = cctk.CTkLabel(self.left_panel, text="Pilih Dataset:", font=cctk.CTkFont(size=13, weight="bold"))
        self.lbl_dataset.pack(anchor="w", padx=25, pady=(10, 2))
        
        self.dataset_menu = cctk.CTkOptionMenu(self.left_panel, values=list(DATASETS.keys()), height=35)
        self.dataset_menu.pack(fill="x", padx=25, pady=5)
        if len(DATASETS) > 0: 
            self.dataset_menu.set(list(DATASETS.keys())[0])

        # Tombol Unggah CSV Baru
        self.btn_add = cctk.CTkButton(
            self.left_panel, 
            text="+ Tambah Dataset CSV Baru", 
            fg_color="transparent", 
            border_width=1, 
            height=32, 
            command=self.add_dataset
        )
        self.btn_add.pack(fill="x", padx=25, pady=(5, 15))

        # Dropdown Menu - Pilih Metode Kombinasi
        self.lbl_method = cctk.CTkLabel(self.left_panel, text="Pilih Kombinasi Metode:", font=cctk.CTkFont(size=13, weight="bold"))
        self.lbl_method.pack(anchor="w", padx=25, pady=(10, 2))

        self.method_menu = cctk.CTkOptionMenu(self.left_panel, values=METHODS, height=35)
        self.method_menu.pack(fill="x", padx=25, pady=5)
        self.method_menu.set(METHODS[0])

        # # Informasi Teknis Pembagian Data
        # self.lbl_info = cctk.CTkLabel(self.left_panel, text="Konfigurasi Validasi:", font=cctk.CTkFont(size=13, weight="bold"))
        # self.lbl_info.pack(anchor="w", padx=25, pady=(20, 2))

        # info_text = "• Pembagian Data: 80% Training, 20% Testing\n• Validasi Silang: 5-Fold Stratified CV\n• Fitur Text Maksimal: 4000 Fitur Token\n• Filter Bahasa: Inggris (Stop Words)"
        # self.txt_info = cctk.CTkLabel(self.left_panel, text=info_text, justify="left", font=cctk.CTkFont(size=11))
        # self.txt_info.pack(anchor="w", padx=30, pady=5)

        # Progress Bar & Status (Indikator Loading Animasi)
        self.progress_bar = cctk.CTkProgressBar(self.left_panel, orientation="horizontal", height=10)
        self.progress_bar.pack(fill="x", padx=25, pady=(25, 0))
        self.progress_bar.set(0)
        
        self.lbl_status = cctk.CTkLabel(self.left_panel, text="Status: Siap", font=cctk.CTkFont(size=11, slant="italic"))
        self.lbl_status.pack(anchor="w", padx=25, pady=(2, 10))

        # Tombol Eksekusi
        self.btn_run = cctk.CTkButton(
            self.left_panel, 
            text="Jalankan Pelatihan & Tes", 
            font=cctk.CTkFont(weight="bold"), 
            height=42, 
            command=self.start_experiment_thread
        )
        self.btn_run.pack(fill="x", padx=25, pady=(10, 20))

        # ==========================================
        # PANEL KANAN (HASIL OUTPUT)
        # ==========================================
        self.right_panel = cctk.CTkFrame(root, corner_radius=15)
        self.right_panel.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        self.right_panel.grid_rowconfigure(1, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

        self.result_title = cctk.CTkLabel(self.right_panel, text="Hasil Evaluasi Eksperimen", font=cctk.CTkFont(size=18, weight="bold"))
        self.result_title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        # Box Monitor Log/Output teks
        self.result_box = cctk.CTkTextbox(self.right_panel, font=cctk.CTkFont(family="Courier", size=13), corner_radius=10)
        self.result_box.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.result_box.insert("1.0", "Silakan atur kombinasi di panel kiri, lalu klik 'Jalankan Pelatihan & Tes'...")

    def add_dataset(self):
        file_path = filedialog.askopenfilename(title="Pilih Dataset CSV", filetypes=[("CSV Files", "*.csv")])
        if not file_path: 
            return
        try:
            import pandas as pd
            df = pd.read_csv(file_path, nrows=5)
            columns = [col.lower().strip() for col in df.columns]
            if not all(col in columns for col in ["subject", "body", "label"]):
                messagebox.showerror("Format Salah", "File CSV wajib memiliki kolom: subject, body, dan label.")
                return
                
            dataset_name = os.path.splitext(os.path.basename(file_path))[0]
            DATASETS[dataset_name] = file_path
            self.dataset_menu.configure(values=list(DATASETS.keys()))
            self.dataset_menu.set(dataset_name)
            messagebox.showinfo("Berhasil", f"Dataset '{dataset_name}' sukses didaftarkan!")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat berkas: {str(e)}")

    def start_experiment_thread(self):
        """Menjalankan eksperimen di thread terpisah agar aplikasi GUI tidak hang/freeze"""
        self.btn_run.configure(state="disabled", text="Memproses...")
        self.progress_bar.configure(mode="indefinite")
        self.progress_bar.start()
        self.lbl_status.configure(text="Status: Memproses data & melatih model...")
        
        self.result_box.delete("1.0", tk.END)
        self.result_box.insert("1.0", "Sedang mengekstrak fitur teks, melakukan validasi silang (5-Fold CV),\ndan mengevaluasi performa model final... Mohon tunggu beberapa saat.")
        
        # Inisialisasi Threading
        threading.Thread(target=self.run_test, daemon=True).start()

    def run_test(self):
        try:
            selected_dataset = self.dataset_menu.get()
            dataset_path = DATASETS[selected_dataset]
            selected_method = self.method_menu.get()

            # Memanggil fungsi worker utama dengan membawa parameter
            res = run_experiment(dataset_path, selected_method)

            # Menyusun struktur representasi tabel matriks evaluasi
            cv_acc_pct = f"{res['cv_metrics']['accuracy']*100:.2f}%"
            cv_prec_pct = f"{res['cv_metrics']['precision']*100:.2f}%"
            cv_rec_pct = f"{res['cv_metrics']['recall']*100:.2f}%"
            cv_f1_pct = f"{res['cv_metrics']['f1']*100:.2f}%"

            test_acc_pct = f"{res['test_metrics']['accuracy']*100:.2f}%"
            test_prec_pct = f"{res['test_metrics']['precision']*100:.2f}%"
            test_rec_pct = f"{res['test_metrics']['recall']*100:.2f}%"
            test_f1_pct = f"{res['test_metrics']['f1']*100:.2f}%"

            # Format teks luaran visual log yang rapi dan terstruktur dalam bentuk tabel komparasi
            output_text = (
                f"{'='*68}\n"
                f"        LOG EKSPERIMEN KLASIFIKASI EMAIL PHISHING\n"
                f"{'='*68}\n"
                f" Nama Dataset : {selected_dataset}\n"
                f" Kombinasi    : {selected_method}\n"
                f" Target Class : {', '.join(res['classes'])}\n"
                f"{'='*68}\n\n"
                f" 1. TABEL METRIKS EVALUASI (DESIMAL DAN PERSENTASE):\n"
                f" +-------------------+---------------------+---------------------+\n"
                f" | Metrik Evaluasi   | 5-Fold Cross Val    | Final Test Set      |\n"
                f" +-------------------+---------------------+---------------------+\n"
                f" | Accuracy          | {res['cv_metrics']['accuracy']:.4f} ({cv_acc_pct:<7}) | {res['test_metrics']['accuracy']:.4f} ({test_acc_pct:<7}) |\n"
                f" | Precision (Macro) | {res['cv_metrics']['precision']:.4f} ({cv_prec_pct:<7}) | {res['test_metrics']['precision']:.4f} ({test_prec_pct:<7}) |\n"
                f" | Recall (Macro)    | {res['cv_metrics']['recall']:.4f} ({cv_rec_pct:<7}) | {res['test_metrics']['recall']:.4f} ({test_rec_pct:<7}) |\n"
                f" | F1-Score (Macro)  | {res['cv_metrics']['f1']:.4f} ({cv_f1_pct:<7}) | {res['test_metrics']['f1']:.4f} ({test_f1_pct:<7}) |\n"
                f" +-------------------+---------------------+---------------------+\n\n"
                f"{'-'*68}\n"
                f" Status Berhasil: Seluruh tahapan pengujian selesai secara optimal.\n"
                f"{'='*68}\n"
            )
            
            # Update Tampilan GUI utama kembali ke thread asal
            self.root.after(0, lambda: self.update_gui_success(output_text))

        except Exception as e:
            self.root.after(0, lambda: self.update_gui_error(str(e)))

    def update_gui_success(self, text):
        self.result_box.delete("1.0", tk.END)
        self.result_box.insert(tk.END, text)
        self.progress_bar.stop()
        self.progress_bar.set(1.0)
        self.lbl_status.configure(text="Status: Selesai diproses.")
        self.btn_run.configure(state="normal", text="Jalankan Pelatihan & Tes")

    def update_gui_error(self, error_msg):
        self.result_box.delete("1.0", tk.END)
        self.result_box.insert("1.0", "Terjadi kesalahan saat mengeksekusi model klasifikasi.")
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.lbl_status.configure(text="Status: Terjadi kesalahan.")
        self.btn_run.configure(state="normal", text="Jalankan Pelatihan & Tes")
        messagebox.showerror("Eksperimen Gagal", error_msg)

if __name__ == "__main__":
    root = cctk.CTk()
    app = App(root)
    root.mainloop()
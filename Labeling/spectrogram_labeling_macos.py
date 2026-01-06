import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import pandas as pd
import os
import pygame


class ImageGridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Revisor de Espectrogramas")

        self.rows = 4
        self.columns = 8
        self.page = 0

        self.selected_label = None
        self.selected_cells = {}
        self.image_refs = []

        self.audio_path = "audio_files/raw_segm_audios"
        pygame.init()
        pygame.mixer.init()

        self.current_audio = None

        # ================= CSV =================
        self.csv_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")]
        )
        if not self.csv_path:
            messagebox.showerror("Error", "No se seleccionó archivo CSV.")
            root.destroy()
            return

        self.starting_page = simpledialog.askinteger(
            "Página inicial",
            "Selecciona la página inicial (0 para la primera):",
            initialvalue=0
        )
        if self.starting_page is not None:
            self.page = self.starting_page

        self.df = pd.read_csv(self.csv_path)
        if "categoria" not in self.df.columns:
            self.df["categoria"] = ""

        # ================= MENÚ =================
        self.menu_frame = tk.Frame(root)
        self.menu_frame.pack(pady=2)

        self.categories = ['no coincide', 'silencio', 'ruido', 'experto', 'voz']
        for cat in self.categories:
            tk.Button(
                self.menu_frame,
                text=cat,
                command=lambda c=cat: self.set_category(c)
            ).pack(side=tk.LEFT, padx=3)

        # ================= KEYBINDS =================
        root.bind("g", lambda e: self.save())
        root.bind("<space>", lambda e: self.next_page())
        root.bind("r", lambda e: self.return_past_page())
        root.bind("c", lambda e: self.play_audio(self.current_audio))
        root.bind("v", lambda e: self.stop_audio())

        total_pages = (len(self.df) - 1) // (self.rows * self.columns) + 1
        self.page_label = tk.Label(
            root,
            text=f"Página {self.page + 1} de {total_pages}",
            font=("Arial", 8)
        )
        self.page_label.pack(pady=3)

        self.grid_frame = tk.Frame(root)
        self.grid_frame.pack(pady=10)

        self.display_grid()

    # =====================================================
    def display_grid(self):
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        self.selected_cells.clear()
        self.image_refs.clear()

        start = self.page * self.rows * self.columns
        end = min(start + self.rows * self.columns, len(self.df))

        for i in range(self.rows):
            for j in range(self.columns):
                idx = start + i * self.columns + j
                if idx >= end:
                    continue

                img_path = str(self.df.iloc[idx, 0])

                try:
                    img = Image.open(img_path)
                    img = img.resize((150, 150))
                    img_tk = ImageTk.PhotoImage(img)
                    self.image_refs.append(img_tk)

                    label = tk.Label(
                        self.grid_frame,
                        image=img_tk,
                        borderwidth=1,
                        relief="solid",
                        highlightthickness=0
                    )
                    label.grid(row=i * 2, column=j, padx=2, pady=2)

                    label.bind(
                        "<Button-1>",
                        lambda e, ix=idx, l=label: self.select_cell(ix, l)
                    )

                    folder = os.path.basename(os.path.dirname(img_path))
                    filename = os.path.basename(img_path)
                    text_ = f"{folder}/{filename}"

                    tk.Label(
                        self.grid_frame,
                        text=text_,
                        font=("Arial", 7)
                    ).grid(row=i * 2 + 1, column=j)

                except Exception as e:
                    print(f"Error cargando imagen {img_path}: {e}")

    # =====================================================
    def select_cell(self, idx, label):
        if idx in self.selected_cells:
            label.config(highlightthickness=0)
            del self.selected_cells[idx]
        else:
            label.config(
                highlightbackground="red",
                highlightcolor="red",
                highlightthickness=3
            )
            self.selected_cells[idx] = label
            self.current_audio = idx

    def set_category(self, category):
        self.selected_label = category
        print(f"Categoría seleccionada: {category}")

    # =====================================================
    def save(self):
        if not self.selected_label:
            messagebox.showwarning(
                "Advertencia",
                "Selecciona una categoría antes de guardar."
            )
            return

        for idx, label in self.selected_cells.items():
            self.df.at[idx, "categoria"] = self.selected_label
            label.config(highlightthickness=0)

        self.selected_cells.clear()
        self.selected_label = None
        self.df.to_csv(self.csv_path, index=False)

        messagebox.showinfo("Guardado", "Categorías guardadas correctamente.")

    # =====================================================
    def return_past_page(self):
        for label in self.selected_cells.values():
            label.config(highlightthickness=0)
        self.selected_cells.clear()

        if self.page > 0:
            self.page -= 1
            self.display_grid()
            total_pages = (len(self.df) - 1) // (self.rows * self.columns) + 1
            self.page_label.config(
                text=f"Página {self.page + 1} de {total_pages}"
            )

    def next_page(self):
        max_pages = (len(self.df) - 1) // (self.rows * self.columns)
        if self.page < max_pages:
            self.page += 1
            self.display_grid()
            total_pages = (len(self.df) - 1) // (self.rows * self.columns) + 1
            self.page_label.config(
                text=f"Página {self.page + 1} de {total_pages}"
            )

    # =====================================================
    def play_audio(self, idx):
        if idx is None:
            return

        img_path = str(self.df.iloc[idx, 0])
        audio_path = img_path.replace(
            "labeling_files/raw_specs",
            self.audio_path
        )

        if "_voice" in audio_path:
            audio_path = audio_path.replace("_voice.png", ".ogg")
        else:
            audio_path = audio_path.replace(".png", ".ogg")

        if os.path.exists(audio_path):
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            print(f"Reproduciendo: {audio_path}")
        else:
            print(f"Audio no encontrado: {audio_path}")

    def stop_audio(self):
        pygame.mixer.music.stop()


# =====================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGridApp(root)
    root.mainloop()
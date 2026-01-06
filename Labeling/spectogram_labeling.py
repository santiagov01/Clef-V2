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
        self.current_audio = None  # Para rastrear el audio actual

        # Cargar CSV
        self.csv_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not self.csv_path:
            messagebox.showerror("Error", "No se seleccionó archivo CSV.")
            # en lugar de cerrar la ventana
            root.destroy()
            return

        # Seleccionar página inicial en caso de que no sea 0
        self.starting_page = None
        self.starting_page = simpledialog.askinteger("Página inicial", "Selecciona la página inicial (0 para la primera):", initialvalue=0)
        if self.starting_page is not None:
            self.page = self.starting_page

        self.df = pd.read_csv(self.csv_path)
        if 'categoria' not in self.df.columns:
            self.df['categoria'] = ""

        # Menú de categorías
        self.menu_frame = tk.Frame(root)
        self.menu_frame.pack(pady=2)

        self.categories = ['no coincide', 'silencio', 'ruido', 'experto', 'voz']
        for cat in self.categories:
            b = tk.Button(self.menu_frame, text=cat, command=lambda c=cat: self.set_category(c))
            b.pack(side=tk.LEFT, padx=3)

        # # Menú de control
        # self.control_frame = tk.Frame(root)
        # self.control_frame.pack()

        root.bind('g', lambda event: self.save()) #guardar etiqueta seleccionada
        root.bind('<space>', lambda event: self.next_page()) #siguiente página
        root.bind('r', lambda event: self.return_past_page()) #volver a la página anterior

        # Asignar teclas para reproducir y detener audio
        root.bind('c', lambda event: self.play_audio(self.current_audio)) #reproducir audio
        root.bind('v', lambda event: self.stop_audio()) #detener audio

        # Añadir cuadro de texto que diga el numero de pagina actual y el total de paginas
        total_pages = (len(self.df) - 1) // (self.rows * self.columns) + 1
        self.page_label = tk.Label(root, text=f"Página {self.page + 1} de {total_pages}", font=("Arial", 8))
        self.page_label.pack(pady=3)

        # Cuadrícula de imágenes
        self.grid_frame = tk.Frame(root)
        self.grid_frame.pack(pady=10)

        self.display_grid()

    def display_grid(self):
        # Limpiar cuadrícula anterior
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        self.cells = []
        self.image_refs = []

        start = self.page * self.rows * self.columns
        end = min(start + self.rows * self.columns, len(self.df))

        # Añadir bajo cada imagen un label muy pequeño con el nombre de la imgen
        for i in range(self.rows):
            row = []
            for j in range(self.columns):
                idx = start + i * self.columns + j
                if idx < end:
                    path = self.df.iloc[idx, 0]
                    try:
                        img = Image.open(path)
                        img = img.resize((150, 150))
                        img_tk = ImageTk.PhotoImage(img)
                        self.image_refs.append(img_tk)  # prevenir recolección
                        label = tk.Label(self.grid_frame, image=img_tk, borderwidth=0.5, relief="solid")
                        label.grid(row=i, column=j, padx=2, pady=2)
                        label.bind("<Button-1>", lambda e, ix=idx, l=label: self.select_cell(ix, l))
                        # Añadir label con el nombre de la imagen
                        filename = os.path.basename(path)
                        filename = os.path.splitext(filename)[0]
                        text_ = path.split("\\")[-2] + '/' + path.split("\\")[-1]
                        # añadirlo
                        label_name = tk.Label(self.grid_frame, text=text_, font=("Arial", 7), bg="white")
                        label_name.grid(row=i, column=j, sticky="s")

                        row.append(label)
                    except Exception as e:
                        print(f"Error cargando imagen {path}: {e}")
                        row.append(None)
                else:
                    row.append(None)
            self.cells.append(row)

    def select_cell(self, idx, label):
        if idx in self.selected_cells:
            # Deseleccionar
            # label.config(bg="SystemButtonFace")
            # del self.selected_cells[idx]
            label.config(highlightbackground="black", highlightthickness=0)
            del self.selected_cells[idx]
        else:
            # label.config(bg="yellow")
            # self.selected_cells[idx] = label
            label.config(highlightbackground="red", highlightthickness=3)
            self.selected_cells[idx] = label
            self.current_audio = idx  # Guardar el índice del audio seleccionado

            #self.play_audio(idx)

    def set_category(self, category):
        self.selected_label = category
        print(f"Categoría seleccionada: {category}")

    def save(self):
        if not self.selected_label:
            messagebox.showwarning("Advertencia", "Selecciona una categoría antes de guardar.")
            return

        for idx in self.selected_cells:
            # self.df.at[idx, 'categoria'] = self.selected_label
            # self.selected_cells[idx].config(bg="SystemButtonFace")
            self.df.at[idx, 'categoria'] = self.selected_label
            self.selected_cells[idx].config(highlightbackground="black", highlightthickness=0)

        self.selected_cells.clear()
        self.selected_label = None
        self.df.to_csv(self.csv_path, index=False)
        messagebox.showinfo("Guardado", "Categorías guardadas correctamente.")

    def return_past_page(self):
        # reset todas las variables de cell seleccionadas
        for idx in self.selected_cells:
            self.selected_cells[idx].config(highlightbackground="black", highlightthickness=0)
        self.selected_cells.clear()
        if self.page > 0:
            self.page -= 1
            self.display_grid()
            # Update page label
            total_pages = (len(self.df) - 1) // (self.rows * self.columns) + 1
            self.page_label.config(text=f"Página {self.page + 1} de {total_pages}")

        else:
            messagebox.showinfo("Inicio", "Ya estás en la primera página.")

    def next_page(self):
        max_pages = (len(self.df) - 1) // (self.rows * self.columns)
        if self.page < max_pages:
            self.page += 1
            self.display_grid()

            # Update page label
            total_pages = (len(self.df) - 1) // (self.rows * self.columns) + 1
            self.page_label.config(text=f"Página {self.page + 1} de {total_pages}")

        else:
            messagebox.showinfo("Fin", "No hay más imágenes.")
            # Renormbrar el archivo .csv que leyó, con el mismo nombre pero con '_done' al final
            csv_path = self.csv_path
            csv_path_done = os.path.splitext(csv_path)[0] + '_done.csv'
            try:
                os.rename(csv_path, csv_path_done)
                messagebox.showinfo("Renombrado", f"Archivo renombrado a: {os.path.basename(csv_path_done)}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo renombrar el archivo: {e}")
    def play_audio(self, idx):
        """
        Reproduce el archivo de audio correspondiente a la imagen seleccionada.
        """
        if idx is None:
            print("No se ha seleccionado ningún audio.")
            return
        # Obtener la ruta del archivo de audio
        image_path = self.df.iloc[idx, 0]
        image_path = image_path.replace("labeling_files/raw_specs", self.audio_path)
        if "_voice" in image_path:
            audio_path = image_path.replace("_voice.png", ".ogg")
        else:
            audio_path = image_path.replace(".png", ".ogg")
        

        if os.path.exists(audio_path):
            try:
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                print(f"Reproduciendo: {audio_path}")
            except Exception as e:
                print(f"Error al reproducir el audio {audio_path}: {e}")
        else:
            print(f"Archivo de audio no encontrado: {audio_path}")

    def stop_audio(self):
        """
        Detiene la reproducción del audio actual.
        """
        pygame.mixer.music.stop()
        print("Audio detenido.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGridApp(root)
    root.mainloop() 

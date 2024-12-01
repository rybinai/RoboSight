import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import cv2
import random
from module_mobile_object import load_mobile_models
from terrain_module import TerrainModelLoader
import static_object_detection

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Видеообработка")
        self.root.geometry("1000x600")

        # Создаем фреймы для разделения интерфейса
        self.left_frame = tk.Frame(root, width=200, height=600, bg="lightgrey")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.right_frame = tk.Frame(root, width=800, height=600)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Кнопки управления
        self.mobile_button = tk.Button(self.left_frame, text="Мобильные объекты", command=self.select_mobile_video, font=("Arial", 14), bg="blue", fg="white")
        self.mobile_button.pack(padx=20, pady=50)

        self.static_button = tk.Button(self.left_frame, text="Статичные объекты", command=self.select_static_video, font=("Arial", 14), bg="green", fg="white")
        self.static_button.pack(padx=20, pady=10)

        self.terrain_button = tk.Button(self.left_frame, text="Распознание рельефа", command=self.select_terrain_video, font=("Arial", 14), bg="orange", fg="white")
        self.terrain_button.pack(padx=20, pady=10)

        self.exit_button = tk.Button(self.left_frame, text="Выход", command=root.destroy, font=("Arial", 14), bg="red", fg="white")
        self.exit_button.pack(side=tk.BOTTOM, padx=20, pady=10)

        self.canvas = tk.Canvas(self.right_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Загрузка моделей
        self.video_processor, self.merger = load_mobile_models()
        
        # Создание экземпляра TerrainModelLoader
        self.terrain_processor = TerrainModelLoader()

        self.running = False

    def select_terrain_video(self):
        video_path = filedialog.askopenfilename(title="Выберите видео для распознавания рельефа", filetypes=[("Видео файлы", "*.mp4 *.avi")])
        if video_path:
            threading.Thread(target=self.process_terrain_video, args=(video_path,)).start()

    def process_terrain_video(self, video_path):
        """Обработка рельефа через terrain_module."""
        video_processor = self.terrain_processor.get_video_processor()
        video_processor.start_video_stream(video_path, self.canvas, self.root)

    def select_mobile_video(self):
        video_path = filedialog.askopenfilename(title="Выберите видео для мобильных объектов", filetypes=[("Видео файлы", "*.mp4 *.avi")])
        if video_path:
            self.running = True
            threading.Thread(target=self.process_mobile_video, args=(video_path,)).start()

    def process_mobile_video(self, video_path):
        """Обработка мобильных объектов через module_mobile_object."""
        video_processor, merger = load_mobile_models()
        video_processor.process_video(video_path, self.canvas, self.root)


    def select_static_video(self):
        video_path = filedialog.askopenfilename(title="Выберите видео для статичных объектов", 
                                                filetypes=[("Видео файлы", "*.mp4 *.avi")])
        if video_path:
            self.running = True
            threading.Thread(target=self.process_static_video, args=(video_path,)).start()


    def process_static_video(self, video_path):
        """Обработка статичных объектов через static_object_detection."""
        static_object_detection.start_static_object_detection(video_path, self.canvas, self.root)

# Блок для запуска интерфейса
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()

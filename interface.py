import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading    
import cv2
from module_mobile_object import load_mobile_models
from terrain_module import TerrainModelLoader
import static_object_detection

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Видеообработка")
        self.root.geometry("270x600")

        # Темный фон для окна
        self.root.config(bg="#2E2E2E")

        # Создаем фреймы для разделения интерфейса
        self.left_frame = tk.Frame(root, width=200, height=600, bg="#2E2E2E")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Кнопки управления
        button_width = 20
        button_height = 2

        self.mobile_button = tk.Button(
            self.left_frame, text="Мобильные объекты", command=self.select_mobile_video,
            font=("Arial", 14), bg="grey", fg="white", width=button_width, height=button_height
        )
        self.mobile_button.pack(side=tk.TOP, padx=20, pady=20)

        self.static_button = tk.Button(
            self.left_frame, text="Статичные объекты", command=self.select_static_video,
            font=("Arial", 14), bg="grey", fg="white", width=button_width, height=button_height
        )
        self.static_button.pack(side=tk.TOP, padx=20, pady=10)

        self.terrain_button = tk.Button(
            self.left_frame, text="Распознание рельефа", command=self.select_terrain_video,
            font=("Arial", 14), bg="grey", fg="white", width=button_width, height=button_height
        )
        self.terrain_button.pack(side=tk.TOP, padx=20, pady=10)

        self.exit_button = tk.Button(
            self.left_frame, text="Выход", command=root.destroy,
            font=("Arial", 14), bg="grey", fg="white", width=button_width, height=button_height
        )
        self.exit_button.pack(side=tk.BOTTOM, padx=20, pady=10)

        # Инициализация моделей как None (загружаются позже)
        self.video_processor = None
        self.merger = None
        self.terrain_processor = None
        self.running = False

    def open_video_window(self, process_func, video_path):
        """Открыть новое окно для видеопотока."""
        video_window = tk.Toplevel(self.root)
        video_window.title("Видеопоток")
        video_window.geometry("800x600")
        video_window.config(bg="black")

        # Холст для отображения видео
        video_canvas = tk.Canvas(video_window, bg="black", width=800, height=600)
        video_canvas.pack(fill=tk.BOTH, expand=True)

        # Кнопка остановки потока
        def stop_video():
            self.running = False
            video_window.destroy()

        close_button = tk.Button(
            video_window, text="Выход", command=stop_video,
            font=("Arial", 14), bg="grey", fg="white"
        )
        close_button.pack(side=tk.BOTTOM, pady=10)

        # Запуск видеопотока
        self.running = True
        threading.Thread(target=process_func, args=(video_path, video_canvas, video_window)).start()

    def select_mobile_video(self):
        """Выбор видео для обработки мобильных объектов."""
        # Загружаем модель, если она еще не загружена
        if self.video_processor is None or self.merger is None:
            print("Загрузка модели для мобильных объектов...")
            self.video_processor, self.merger = load_mobile_models()
            print("Модель успешно загружена.")
        
        # Открываем диалоговое окно для выбора видео
        video_path = filedialog.askopenfilename(
            title="Выберите видео для мобильных объектов",
            filetypes=[("Видео файлы", "*.mp4 *.avi")]
        )
        if video_path:
            self.open_video_window(self.process_mobile_video, video_path)

    def process_mobile_video(self, video_path, canvas, window):
        """Обработка мобильных объектов через module_mobile_object.py."""
        try:
            if self.video_processor is None:
                raise Exception("Ошибка: Модель не загружена.")
            
            # Используем VideoProcessor из module_mobile_object
            self.video_processor.process_video(video_path, canvas, window)
        except Exception as e:
            print(f"Ошибка обработки видео: {e}")
            window.destroy()

    def select_static_video(self):
        """Выбор видео для обработки статичных объектов."""
        video_path = filedialog.askopenfilename(
            title="Выберите видео для статичных объектов",
            filetypes=[("Видео файлы", "*.mp4 *.avi")]
        )
        if video_path:
            self.open_video_window(self.process_static_video, video_path)

    def process_static_video(self, video_path, canvas, window):
        """Обработка статичных объектов."""
        static_object_detection.start_static_object_detection(video_path, canvas, window)

    def select_terrain_video(self):
        """Выбор видео для обработки рельефа."""
        # Загружаем модель, если она еще не загружена
        if self.terrain_processor is None:
            print("Загрузка модели для распознавания рельефа...")
            self.terrain_processor = TerrainModelLoader()
            print("Модель для рельефа успешно загружена.")
        
        video_path = filedialog.askopenfilename(
            title="Выберите видео для распознавания рельефа",
            filetypes=[("Видео файлы", "*.mp4 *.avi")]
        )
        if video_path:
            self.open_video_window(self.process_terrain_video, video_path)

    def process_terrain_video(self, video_path, canvas, window):
        """Обработка рельефа."""
        video_processor = self.terrain_processor.get_video_processor()
        video_processor.start_video_stream(video_path, canvas, window)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
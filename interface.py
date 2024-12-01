import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import cv2
from ultralytics import YOLO
import random
from terrain_module import RealTimeVideoProcessor 
import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from module_mobile_object import VideoProcessor, DetectionMerger
import static_object_detection  # Импорт для обработки статичных объектов

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Видеообработка с YOLO")
        self.root.geometry("1000x600")

        # Создаем фреймы для разделения интерфейса
        self.left_frame = tk.Frame(root, width=200, height=600, bg="lightgrey")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.right_frame = tk.Frame(root, width=800, height=600)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Кнопки управления слева
        self.mobile_button = tk.Button(self.left_frame, text="Мобильные объекты", command=self.select_mobile_video, 
                                       font=("Arial", 14), bg="blue", fg="white")
        self.mobile_button.pack(padx=20, pady=50)

        self.static_button = tk.Button(self.left_frame, text="Статичные объекты", command=self.select_static_video, 
                                       font=("Arial", 14), bg="green", fg="white")
        self.static_button.pack(padx=20, pady=10)

        self.terrain_button = tk.Button(self.left_frame, text="Распознание рельефа", command=self.select_terrain_video, font=("Arial", 14), bg="orange", fg="white")
        self.terrain_button.pack(padx=20, pady=10)

        self.exit_button = tk.Button(self.left_frame, text="Выход", command=root.destroy, 
                                     font=("Arial", 14), bg="red", fg="white")
        self.exit_button.pack(side=tk.BOTTOM, padx=20, pady=10)

        # Холст для отображения видео
        self.canvas = tk.Canvas(self.right_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Настройка моделей
        self.models = [
            YOLO('D:/USER/Desktop/studies/python/main/temp/fox.pt'),
            YOLO('D:/USER/Desktop/studies/python/main/temp/people.pt'),
            YOLO('D:/USER/Desktop/studies/python/main/temp/rabbit.pt')
        ]
        for model in self.models:
            model.fuse()

        self.merger = DetectionMerger(iou_threshold=0.5)
        self.video_processor = VideoProcessor(self.models, self.merger, show_video=False, save_video=False)

        self.running = False

        # Инициализация модели DeeplabV3 для рельефа
        self.terrain_model = deeplabv3_mobilenet_v3_large(num_classes=7)
        self.terrain_model.load_state_dict(torch.load("D:/USER/Desktop/studies/python/main/temp/model_win_10.pth", map_location=torch.device('cpu')))
        self.terrain_model.eval()
        self.terrain_processor = RealTimeVideoProcessor(self.terrain_model)
    
    def select_terrain_video(self):
        video_path = filedialog.askopenfilename(title="Выберите видео для распознавания рельефа", filetypes=[("Видео файлы", "*.mp4 *.avi")])
        if video_path:
            threading.Thread(target=self.terrain_processor.start_video_stream, args=(video_path, self.canvas, self.root)).start()

    def select_mobile_video(self):
        video_path = filedialog.askopenfilename(title="Выберите видео для мобильных объектов", 
                                                filetypes=[("Видео файлы", "*.mp4 *.avi")])
        if video_path:
            self.running = True
            threading.Thread(target=self.process_mobile_video, args=(video_path,)).start()

    def select_static_video(self):
        video_path = filedialog.askopenfilename(title="Выберите видео для статичных объектов", 
                                                filetypes=[("Видео файлы", "*.mp4 *.avi")])
        if video_path:
            self.running = True
            threading.Thread(target=self.process_static_video, args=(video_path,)).start()

    def process_mobile_video(self, video_path):
        """Обработка мобильных объектов."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Error: Could not open video file.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            all_detections = []
            for model in self.models:
                results = model.track(frame, iou=0.4, conf=0.5, persist=True, imgsz=608, verbose=False)
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    scores = results[0].boxes.conf.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    for box, score, id in zip(boxes, scores, ids):
                        x1, y1, x2, y2 = box
                        all_detections.append([x1, y1, x2, y2, score, id])
            merged_detections = self.merger.merge_detections(all_detections)
            for detection in merged_detections:
                x1, y1, x2, y2, score, obj_id = detection
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Id {obj_id} | Conf: {score:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))
            img = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()

    def process_static_video(self, video_path):
        """Обработка статичных объектов через static_object_detection."""
        static_object_detection.start_static_object_detection(video_path, self.canvas, self.root)


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
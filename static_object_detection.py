import cv2
import random
import threading
from PIL import Image, ImageTk
from ultralytics import YOLO
from pathlib import Path
import logging

logging.getLogger('ultralytics').setLevel(logging.WARNING)

class ObjectDetectionProcessor:
    def __init__(self, models, labels, input_video_path, canvas, root, output_size=(800, 600)):
        self.models = models
        self.labels = labels
        self.input_video_path = input_video_path
        self.canvas = canvas
        self.root = root
        self.output_size = output_size  # Размер отображаемого видео

        self.cap = cv2.VideoCapture(input_video_path)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open video file.")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def process_video(self):
        # Обработка видео с использованием моделей и вывод на экран.
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Получаем детекции для текущего кадра
            all_detections = self._process_frame(frame)

            # Отрисовка детекций на кадре
            self._draw_detections(frame, all_detections)

            # Преобразуем кадр в RGB для Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.output_size)  # Изменяем размер на указанный пользователем

            # Преобразуем в изображение Tkinter
            img = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))

            # Обновляем холст с использованием after() для синхронизации с главным потоком
            self.update_canvas(img)

            # Пропуск кадров (для улучшения производительности)
            cv2.waitKey(int(1000 / self.fps))  # Фиксируем задержку для управления частотой кадров

            # Выход по нажатию клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self._release_resources()

    def _process_frame(self, frame):
        # Обработка одного кадра и получение всех детекций.
        all_detections = []
        for model, label in zip(self.models, self.labels):
            results = model(frame)  # Использование модели для детекции объектов

            if results[0].boxes is not None:  # Проверка наличия детекций
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confidences):
                    object_label = f"{label}"
                    size = self._calculate_size(box)
                    all_detections.append((box, object_label, conf, size))

        return all_detections

    def _calculate_size(self, box):
        # Вычисление размера объекта по площади бокса.
        width = box[2] - box[0]
        height = box[3] - box[1]
        size = width * height  # Площадь объекта в пикселях
        return size

    def _draw_detections(self, frame, detections):
        # Отрисовка всех детекций на кадре.
        for box, object_label, conf, size in detections:
            random.seed(hash(object_label))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2

            # Отображение метки и размера
            cv2.putText(
                frame,
                f"{object_label} | {conf:.2f} | Size: {size} px",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

    def update_canvas(self, img):
        # Обновление изображения на холсте Tkinter.
        self.img_tk = img
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)
        self.root.update_idletasks()

    def _release_resources(self):
        # Освобождение ресурсов.
        self.cap.release()
        cv2.destroyAllWindows()

def start_static_object_detection(input_video_path, canvas, root, output_size=(800, 600)):
    # Корневая директория проекта
    project_root = Path(__file__).parent.resolve()
    
    # Пути к файлам моделей
    model_paths = [
        project_root / "models" / "static_models" / "tree.pt",
        project_root / "models" / "static_models" / "stone.pt",
        project_root / "models" / "static_models" / "bush.pt"
    ]
    
    # Загружаем модели
    models = [YOLO(str(path)) for path in model_paths]
    labels = ["tree", "stone", "bush"]

    for model in models:
        model.fuse()

    # Передаём размер вывода в объект процессора
    processor = ObjectDetectionProcessor(models, labels, input_video_path, canvas, root, output_size)
    processor.process_video()
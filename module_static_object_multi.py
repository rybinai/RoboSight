import cv2
import random
import threading
import queue
from ultralytics import YOLO

class ObjectDetectionProcessor:
    def __init__(self, models, labels, input_video_path, detection_queue):
        self.models = models
        self.labels = labels
        self.input_video_path = input_video_path
        self.detection_queue = detection_queue  # Очередь для передачи кадров в интерфейс

        self.cap = cv2.VideoCapture(input_video_path)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open video file.")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def process_video(self):
        """Обработка видео с использованием моделей."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Получаем детекции для текущего кадра
            all_detections = self._process_frame(frame)

            # Отрисовка детекций на кадре
            self._draw_detections(frame, all_detections)

            # Отправка кадра в очередь
            if not self.detection_queue.full():
                self.detection_queue.put(frame)

        self._release_resources()

    def _process_frame(self, frame):
        """Обработка одного кадра и получение всех детекций."""
        all_detections = []
        for model, label in zip(self.models, self.labels):
            results = model(frame)  # Использование модели для детекции объектов

            if results[0].boxes is not None:  # Проверка наличия детекций
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confidences):
                    object_label = f"{label}"
                    all_detections.append((box, object_label, conf))

        return all_detections

    def _draw_detections(self, frame, detections):
        """Отрисовка всех детекций на кадре."""
        for box, object_label, conf in detections:
            random.seed(hash(object_label))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2

            cv2.putText(
                frame,
                f"{object_label} | {conf:.2f}",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

    def _release_resources(self):
        """Освобождение ресурсов."""
        self.cap.release()
        cv2.destroyAllWindows()


def run_video_processing(input_video_path, detection_queue):
    # Создание объектов моделей
    models = [
        YOLO('D:/USER/Desktop/studies/python/main/tree.pt'),
        YOLO('D:/USER/Desktop/studies/python/main/stone.pt'),
        YOLO('D:/USER/Desktop/studies/python/main/bush.pt')
    ]
    labels = ["tree", "stone", "bush"]

    # Оптимизация моделей
    for model in models:
        model.fuse()

    # Создание и запуск процессора
    processor = ObjectDetectionProcessor(
        models=models,
        labels=labels,
        input_video_path=input_video_path,
        detection_queue=detection_queue
    )

    processor.process_video()


# Многозадачность для видео обработки
def start_video_processing(input_video_path, detection_queue):
    processing_thread = threading.Thread(target=run_video_processing, args=(input_video_path, detection_queue))
    processing_thread.start()

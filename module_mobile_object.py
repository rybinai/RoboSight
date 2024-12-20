import cv2
from ultralytics import YOLO
import random
import numpy as np
from PIL import Image, ImageTk
import threading
import tkinter as tk
from pathlib import Path

class DetectionMerger:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def merge_detections(self, detections):
        merged_boxes = []
        for detection in detections:
            x1, y1, x2, y2, score, obj_id, class_name = detection
            add_new = True
            for i, merged_box in enumerate(merged_boxes):
                mx1, my1, mx2, my2, mscore, mid, mclass_name = merged_box
                inter_x1 = max(x1, mx1)
                inter_y1 = max(y1, my1)
                inter_x2 = min(x2, mx2)
                inter_y2 = min(y2, my2)
                inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

                box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
                merged_area = (mx2 - mx1 + 1) * (my2 - my1 + 1)
                iou = inter_area / float(box_area + merged_area - inter_area)

                if iou > self.iou_threshold:
                    if score > mscore:
                        merged_boxes[i] = [x1, y1, x2, y2, score, obj_id, class_name]
                    add_new = False
                    break

            if add_new:
                merged_boxes.append([x1, y1, x2, y2, score, obj_id, class_name])
        return merged_boxes


class VideoProcessor:
    def __init__(self, models, merger, show_video=False, save_video=False):
        self.models = models
        self.merger = merger
        self.show_video = show_video
        self.save_video = save_video
        self.previous_positions = {}
        self.previous_timestamps = {}

    def process_video(self, input_video_path, canvas, root):
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise Exception("Error: Could not open video file.")

        prev_frame_shape = None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1 / fps  # Время одного кадра

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Проверка размеров кадров
            if prev_frame_shape is not None and frame.shape[:2] != prev_frame_shape:
                frame = cv2.resize(frame, (prev_frame_shape[1], prev_frame_shape[0]))
            else:
                prev_frame_shape = frame.shape[:2]

            all_detections = []
            for model in self.models:
                results = model.track(frame, iou=0.4, conf=0.7, persist=True, imgsz=608, verbose=False)
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    scores = results[0].boxes.conf.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                    for box, score, obj_id, class_id in zip(boxes, scores, ids, class_ids):
                        x1, y1, x2, y2 = box
                        class_name = model.names[class_id]
                        all_detections.append([x1, y1, x2, y2, score, obj_id, class_name])

            merged_detections = self.merger.merge_detections(all_detections)

            for detection in merged_detections:
                x1, y1, x2, y2, score, obj_id, class_name = detection
                random.seed(int(obj_id))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Вычисление центра объекта
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Вычисление скорости
                if obj_id in self.previous_positions:
                    prev_x, prev_y = self.previous_positions[obj_id]
                    distance = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
                    speed = distance / frame_time
                else:
                    speed = 0.0

                # Обновление позиции объекта
                self.previous_positions[obj_id] = (center_x, center_y)

                # Отображение информации
                cv2.putText(frame, f"Id {obj_id} | {class_name} | Speed: {speed:.2f} px/s",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Конвертация кадра для tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (800, 600))
            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.image = img

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()


def load_mobile_models():
    # Корневая директория проекта
    project_root = Path(__file__).parent.resolve()

    # Пути к файлам моделей
    model_paths = [
        project_root / "models" / "mobile_models" / "fox.pt",
        project_root / "models" / "mobile_models" / "people.pt",
        project_root / "models" / "mobile_models" / "rabbit.pt"
    ]

    # Загружаем модели
    models = [YOLO(str(path)) for path in model_paths]
    for model in models:
        model.fuse()
    merger = DetectionMerger(iou_threshold=0.5)
    video_processor = VideoProcessor(models, merger, show_video=False, save_video=False)
    return video_processor, merger
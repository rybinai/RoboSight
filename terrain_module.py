import cv2
import torch
from torchvision.transforms import ToTensor, Resize, Normalize
import numpy as np
from PIL import Image, ImageTk

class RealTimeVideoProcessor:
    def __init__(self, model, target_size=(512, 512)):
        self.model = model
        self.target_size = target_size

    def preprocess_frame(self, frame):
        transform = Resize(self.target_size)
        frame_tensor = ToTensor()(frame)
        frame_resized = transform(frame_tensor)
        frame_normalized = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(frame_resized)
        return frame_normalized.unsqueeze(0)

    def apply_colormap(self, mask, palette):
        return palette[mask]

    def postprocess_mask(self, mask, original_size):
        return cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

    def process_frame(self, frame, width, height):
        # Обработка кадра с моделью
        input_tensor = self.preprocess_frame(frame)
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            output = torch.argmax(output, dim=0).cpu().numpy()

        palette = np.random.randint(0, 255, size=(7, 3), dtype=np.uint8)  # Палитра для 7 классов
        color_mask = self.apply_colormap(output, palette)
        segmented_mask = self.postprocess_mask(color_mask, (width, height))
        overlay = cv2.addWeighted(frame, 0.7, segmented_mask, 0.3, 0)

        return overlay

    def update_frame(self, cap, canvas, root):
        ret, frame = cap.read()
        if ret:
            # Обработка кадра
            height, width, _ = frame.shape
            processed_frame = self.process_frame(frame, width, height)

            # Конвертация в изображение для Tkinter
            img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img)

            # Обновление изображения на холсте
            canvas.create_image(0, 0, anchor="nw", image=img_tk)
            canvas.image = img_tk

            # Продолжаем захват видео
            root.after(10, self.update_frame, cap, canvas, root)

    def start_video_stream(self, video_source, canvas, root):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Ошибка при открытии видео потока")
            return

        self.update_frame(cap, canvas, root)
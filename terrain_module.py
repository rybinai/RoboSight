import cv2
import torch
from torchvision.transforms import ToTensor, Resize, Normalize
import numpy as np
from PIL import Image, ImageTk
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from pathlib import Path

class RealTimeVideoProcessor:
    def __init__(self, model, target_size=(512, 512), display_size=(800, 600)):
        self.model = model
        self.target_size = target_size  # Размер для обработки
        self.display_size = display_size  # Размер для отображения

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
        """Обработка кадра с моделью."""
        input_tensor = self.preprocess_frame(frame)
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            output = torch.argmax(output, dim=0).cpu().numpy()

        # палитра цветов
        palette = {
            0: (0, 255, 255),    # Urban land
            1: (255, 255, 0),    # Agriculture land
            2: (255, 0, 255),    # Rangeland
            3: (0, 255, 0),      # Forest land
            4: (255, 0, 64),     # Water (unknown)
            5: (255, 255, 255),  # Barren land
            6: (0, 0, 0)         # Unknown
        }

        color_mask = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
        for class_idx, color in palette.items():
            color_mask[output == class_idx] = color

        segmented_mask = self.postprocess_mask(color_mask, (width, height))
        overlay = cv2.addWeighted(frame, 0.7, segmented_mask, 0.3, 0)

        return overlay

    def update_frame(self, cap, canvas, root):
        ret, frame = cap.read()
        if ret:
            # Приведение кадра к размеру для обработки
            standardized_size = self.target_size
            frame = cv2.resize(frame, standardized_size, interpolation=cv2.INTER_AREA)

            # Обработка кадра
            height, width = standardized_size
            processed_frame = self.process_frame(frame, width, height)

            # Приведение кадра к размеру для отображения
            display_frame = cv2.resize(processed_frame, self.display_size, interpolation=cv2.INTER_AREA)

            # Конвертация в изображение для Tkinter
            img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img)

            # Обновление изображения на холсте
            canvas.create_image(0, 0, anchor="nw", image=img_tk)
            canvas.image = img_tk

            root.after(10, self.update_frame, cap, canvas, root)

    def start_video_stream(self, video_source, canvas, root):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Ошибка при открытии видео потока")
            return

        # Размеры холста для отображения
        canvas.config(width=self.display_size[0], height=self.display_size[1])

        self.update_frame(cap, canvas, root)

class TerrainModelLoader:
    def __init__(self):
        # Определяем корневую директорию проекта
        project_root = Path(__file__).parent.resolve()
        
        # Пути к модели
        model_path = project_root / "models" / "terrain_model" / "terrain.pth"
        
        # Загружаем модель
        self.model = deeplabv3_mobilenet_v3_large(num_classes=7)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
        # Инициализация обработчика видео
        self.video_processor = RealTimeVideoProcessor(self.model)

    def get_video_processor(self):
        # Возвращаем обработчик видео
        return self.video_processor
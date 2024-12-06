import unittest
import tkinter as tk
from module_mobile_object import load_mobile_models, VideoProcessor
from static_object_detection import start_static_object_detection
from terrain_module import TerrainModelLoader
import cv2


class IntegrationTestVideoProcessing(unittest.TestCase):
    def setUp(self):
        """Инициализация среды для тестирования: создание интерфейса и загрузка моделей."""
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=800, height=450)
        self.canvas.pack()
        
        # Загружаем модели для мобильных объектов
        self.mobile_processor, self.mobile_merger = load_mobile_models()
        
        # Создание экземпляра TerrainModelLoader
        self.terrain_loader = TerrainModelLoader()
        
        # Путь к тестовому видеофайлу
        self.test_video_path = "C:/prog/RoboSight/tree1v.mp4" 

    def test_mobile_objects_integration(self):
        """Тест интеграции модуля обработки мобильных объектов."""
        try:
            self.mobile_processor.process_video(self.test_video_path, self.canvas, self.root)
            print("Интеграция мобильных объектов прошла успешно.")
        except Exception as e:
            self.fail(f"Ошибка в модуле мобильных объектов: {e}")\
            
    def test_terrain_segmentation_integration(self):
            """Тест интеграции модуля сегментации рельефа."""
            try:
                video_processor = self.terrain_loader.get_video_processor()
                video_processor.start_video_stream(self.test_video_path, self.canvas, self.root)
                print("Интеграция сегментации рельефа прошла успешно.")
            except Exception as e:
                self.fail(f"Ошибка в модуле сегментации рельефа: {e}")
                
    def test_static_objects_integration(self):
        """Тест интеграции модуля распознавания статичных объектов."""
        try:
            start_static_object_detection(self.test_video_path, self.canvas, self.root)
            print("Интеграция статичных объектов прошла успешно.")
        except Exception as e:
            self.fail(f"Ошибка в модуле статичных объектов: {e}")

    

    def tearDown(self):
        """Закрытие интерфейса после тестирования."""
        self.root.destroy()


if __name__ == "__main__":
    unittest.main()

import sys
import cv2
import queue
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import module_static_object_multi 
#Интерфейс
class VideoDisplayWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoboSight Video Display")
        self.setGeometry(0, 0, 800, 600)

        self.detection_queue = queue.Queue(maxsize=10)  # Очередь для кадров видео
        self.processing_video = False  # Флаг для обработки видео
        self.video_thread = None  # Поток для обработки видео

        # Разворачиваем окно на весь экран, но панель задач не скрывается
        self.showMaximized()

        # Стили для темной темы
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E2E2E;
            }
            QLabel {
                color: white;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #444;
                color: white;
                font-size: 16px;
                padding: 10px;
                border: 1px solid #555;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)

        # Лейауты
        self.main_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        # Надпись RoboSight
        self.title_label = QLabel("RoboSight")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.left_layout.addWidget(self.title_label)

        # Кнопка для начала распознавания
        self.static_objects_btn = QPushButton("Распознавание неподвижных объектов")
        self.left_layout.addWidget(self.static_objects_btn)
        self.static_objects_btn.clicked.connect(self.run_static_objects_detection)

        # Кнопка для остановки распознавания
        self.stop_btn = QPushButton("Остановить распознавание")
        self.left_layout.addWidget(self.stop_btn)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_video_processing)

        # Кнопки для других действий
        self.dynamic_objects_btn = QPushButton("Распознавание мобильных объектов")
        self.terrain_btn = QPushButton("Распознавание рельефа")

        # Стили кнопок
        self.dynamic_objects_btn.setStyleSheet("""
            QPushButton {
                background-color: #444; 
                color: white; 
                font-size: 16px; 
                padding: 10px;
                border: 1px solid #555;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        self.terrain_btn.setStyleSheet("""
            QPushButton {
                background-color: #444; 
                color: white; 
                font-size: 16px; 
                padding: 10px;
                border: 1px solid #555;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)

        self.left_layout.addWidget(self.dynamic_objects_btn)
        self.left_layout.addWidget(self.terrain_btn)
        self.left_layout.addStretch()

        # Подключение кнопок
        self.dynamic_objects_btn.clicked.connect(lambda: self.update_main_view("dynamic"))
        self.terrain_btn.clicked.connect(lambda: self.update_main_view("terrain"))

        # Виджет для отображения видео
        self.video_label = QLabel(self)
        self.right_layout.addWidget(self.video_label)

        # Установка лейаутов
        self.main_layout.addLayout(self.left_layout, 1)
        self.main_layout.addLayout(self.right_layout, 2)
        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        # Таймер для обновления кадров
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def run_static_objects_detection(self):
        """Запуск программы для распознавания неподвижных объектов."""
        if self.processing_video:
            return  # Если видео уже обрабатывается, не запускаем снова

        input_video_path = "D:/USER/Desktop/studies/python/main/tree1v.mp4"  # Путь к видео

        self.processing_video = True
        self.timer.stop()

        # Запуск обработки видео в отдельном потоке
        self.video_thread = threading.Thread(target=self.start_video_processing, args=(input_video_path,))
        self.video_thread.start()

        # Запуск таймера для обновления кадров
        self.timer.start(30)

        # Ожидаем завершения распознавания
        self.stop_btn.setEnabled(True)
        self.static_objects_btn.setEnabled(False)

    def start_video_processing(self, input_video_path):
        """Обработка видео."""
        module_static_object_multi.start_video_processing(input_video_path, self.detection_queue)

    def stop_video_processing(self):
        """Остановка алгоритма распознавания."""
        self.processing_video = False
        self.timer.stop()
        if self.video_thread is not None and self.video_thread.is_alive():
            self.video_thread.join()
        self.stop_btn.setEnabled(False)
        self.static_objects_btn.setEnabled(True)

    def update_frame(self):
        """Обновление видео."""
        if self.processing_video and not self.detection_queue.empty():
            frame = self.detection_queue.get()

            # Преобразование изображения из OpenCV в формат для QLabel
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))

    def update_main_view(self, view_type):
        """Обновление вида в зависимости от типа действия."""
        if view_type == "dynamic":
            self.video_label.setText("Распознавание мобильных объектов")
        elif view_type == "terrain":
            self.video_label.setText("Распознавание рельефа")

    def closeEvent(self, event):
        """При закрытии окна останавливаем видео обработку."""
        if self.processing_video:
            self.stop_video_processing()  # Остановка видео
        event.accept()  # Принять событие закрытия окна


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoDisplayWindow()
    window.showMaximized()  # Окно разворачивается на весь экран, панель задач остается видимой
    sys.exit(app.exec_())

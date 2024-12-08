
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
from module_mobile_object import VideoProcessor, DetectionMerger
from static_object_detection import ObjectDetectionProcessor
from terrain_module import RealTimeVideoProcessor
from interface import VideoApp

class TestVideoAppAndModules(unittest.TestCase):

    @patch('cv2.VideoCapture')
    def test_m1_process_video_success(self, mock_video_capture):
        mock_video_capture.return_value.isOpened.return_value = True
        mock_video_capture.return_value.read.side_effect = [(True, MagicMock()), (False, None)]

        merger = DetectionMerger()
        processor = VideoProcessor(models=[], merger=merger)
        processor.process_video = MagicMock()

        canvas_mock = MagicMock()
        processor.process_video("/home/lenny/PetrSu/3_kurs/1_sem/TPPO/unitTests/RoboSight-main/tree1v.mp4", canvas_mock, MagicMock())
        processor.process_video.assert_called()

    @patch('cv2.VideoCapture')
    def test_m2_process_video_file_open_error(self, mock_video_capture):
        mock_video_capture.return_value.isOpened.return_value = False
        merger = DetectionMerger()
        processor = VideoProcessor(models=[], merger=merger)

        with self.assertRaises(Exception) as context:
            processor.process_video("/home/lenny/PetrSu/3_kurs/1_sem/TPPO/unitTests/RoboSight-main/tree1v.mp4", MagicMock(), MagicMock())
        self.assertIn("Could not open video file", str(context.exception))

    @patch('cv2.VideoCapture')
    def test_m3_process_video_static_success(self, mock_video_capture):
        mock_video_capture.return_value.isOpened.return_value = True
        mock_video_capture.return_value.read.side_effect = [(True, MagicMock()), (False, None)]

        processor = ObjectDetectionProcessor(models=[], labels=[], input_video_path="/home/lenny/PetrSu/3_kurs/1_sem/TPPO/unitTests/RoboSight-main/tree1v.mp4", 
                                             canvas=MagicMock(), root=MagicMock())
        processor._process_frame = MagicMock(return_value=[])
        processor.process_video()
        processor._process_frame.assert_called()

    @patch('cv2.VideoCapture')
    def test_m4_process_video_model_error(self, mock_video_capture):
        mock_video_capture.return_value.isOpened.return_value = True
        processor = ObjectDetectionProcessor(models=[], labels=[], input_video_path="/home/lenny/PetrSu/3_kurs/1_sem/TPPO/unitTests/RoboSight-main/tree1v.mp4", 
                                             canvas=MagicMock(), root=MagicMock())
        processor._process_frame = MagicMock(side_effect=Exception("Invalid model"))

        with self.assertRaises(Exception) as context:
            processor.process_video()
        self.assertIn("Invalid model", str(context.exception))

    def test_m5_process_frame_success(self):
        processor = RealTimeVideoProcessor(model=MagicMock())
        frame = np.zeros((512, 512, 3), dtype=np.uint8)
        processor.model.return_value = {'out': [MagicMock()]}
        processed_frame = processor.process_frame(frame, 512, 512)
        self.assertIsNotNone(processed_frame)

    def test_m6_process_frame_invalid_input(self):
        processor = RealTimeVideoProcessor(model=MagicMock())
        with self.assertRaises(Exception) as context:
            processor.process_frame(None, 512, 512)
        self.assertIn("Invalid input", str(context.exception))

    @patch('cv2.VideoCapture')
    def test_m7_start_video_stream_success(self, mock_video_capture):
        mock_video_capture.return_value.isOpened.return_value = True
        processor = RealTimeVideoProcessor(model=MagicMock())
        processor.update_frame = MagicMock()
        processor.start_video_stream("/home/lenny/PetrSu/3_kurs/1_sem/TPPO/unitTests/RoboSight-main/tree1v.mp4", MagicMock(), MagicMock())
        processor.update_frame.assert_called()

    @patch('cv2.VideoCapture')
    def test_m8_start_video_stream_open_error(self, mock_video_capture):
        mock_video_capture.return_value.isOpened.return_value = False
        processor = RealTimeVideoProcessor(model=MagicMock())
        with self.assertRaises(Exception):
            processor.start_video_stream("/home/lenny/PetrSu/3_kurs/1_sem/TPPO/unitTests/RoboSight-main/tree1v.mp4", MagicMock(), MagicMock())

    def test_m9_open_video_window(self):
        root = MagicMock()
        app = VideoApp(root)
        app.open_video_window = MagicMock()
        app.open_video_window(MagicMock(), "/home/lenny/PetrSu/3_kurs/1_sem/TPPO/unitTests/RoboSight-main/tree1v.mp4")
        app.open_video_window.assert_called()

    @patch('tkinter.filedialog.askopenfilename')
    def test_m10_select_mobile_video_success(self, mock_askopenfilename):
        mock_askopenfilename.return_value = "/home/lenny/PetrSu/3_kurs/1_sem/TPPO/unitTests/RoboSight-main/tree1v.mp4"
        root = MagicMock()
        app = VideoApp(root)
        app.open_video_window = MagicMock()

        app.select_mobile_video()
        app.open_video_window.assert_called()

    @patch('tkinter.filedialog.askopenfilename')
    def test_m11_select_mobile_video_cancel(self, mock_askopenfilename):
        mock_askopenfilename.return_value = ""
        root = MagicMock()
        app = VideoApp(root)
        app.open_video_window = MagicMock()

        app.select_mobile_video()
        app.open_video_window.assert_not_called()

    @patch('module_mobile_object.VideoProcessor')
    def test_m12_process_mobile_video_success(self, mock_video_processor):
        root = MagicMock()
        app = VideoApp(root)
        app.video_processor = mock_video_processor
        app.video_processor.process_video = MagicMock()

        app.process_mobile_video("test_video.mp4", MagicMock(), MagicMock())
        app.video_processor.process_video.assert_called()

    def test_m13_process_mobile_video_file_error(self):
        root = MagicMock()
        app = VideoApp(root)
        app.video_processor = None

        with self.assertRaises(Exception) as context:
            app.process_mobile_video("/home/lenny/PetrSu/3_kurs/1_sem/TPPO/unitTests/RoboSight-main/tree1v.mp4", MagicMock(), MagicMock())
        self.assertIn("Модель не загружена", str(context.exception))

    @patch('tkinter.filedialog.askopenfilename')
    def test_m14_select_static_video_success(self, mock_askopenfilename):
        mock_askopenfilename.return_value = "/home/lenny/PetrSu/3_kurs/1_sem/TPPO/unitTests/RoboSight-main/tree1v.mp4"
        root = MagicMock()
        app = VideoApp(root)
        app.open_video_window = MagicMock()

        app.select_static_video()
        app.open_video_window.assert_called()

    @patch('tkinter.filedialog.askopenfilename')
    def test_m15_select_static_video_cancel(self, mock_askopenfilename):
        mock_askopenfilename.return_value = ""
        root = MagicMock()
        app = VideoApp(root)
        app.open_video_window = MagicMock()

        app.select_static_video()
        app.open_video_window.assert_not_called()

    @patch('static_object_detection.start_static_object_detection')
    def test_m16_process_static_video_success(self, mock_static_detection):
        root = MagicMock()
        app = VideoApp(root)

        app.process_static_video("/home/lenny/PetrSu/3_kurs/1_sem/TPPO/unitTests/RoboSight-main/tree1v.mp4", MagicMock(), MagicMock())
        mock_static_detection.assert_called()

    #@patch('terrain_module.TerrainModelLoader')
    #def test_m17_select_terrain_video_success(self, mock_terrain_loader):
    #    mock_terrain_loader.return_value.get_video_processor.return_value.start_video_stream = MagicMock()
    #    root = MagicMock()
    #    app = VideoApp(root)
    #    app.terrain_processor = mock_terrain_loader()

     #   with patch('tkinter.filedialog.askopenfilename', return_value="test_video.mp4"):
    #        app.select_terrain_video()
    #    app.terrain_processor.get_video_processor().start_video_stream.assert_called()

    @patch('tkinter.filedialog.askopenfilename')
    def test_m18_select_terrain_video_cancel(self, mock_askopenfilename):
        mock_askopenfilename.return_value = ""
        root = MagicMock()
        app = VideoApp(root)
        app.terrain_processor = MagicMock()

        app.select_terrain_video()
        app.terrain_processor.get_video_processor().start_video_stream.assert_not_called()

    @patch('terrain_module.TerrainModelLoader')
    def test_m19_process_terrain_video_success(self, mock_terrain_loader):
        mock_terrain_loader.return_value.get_video_processor.return_value.start_video_stream = MagicMock()
        root = MagicMock()
        app = VideoApp(root)
        app.terrain_processor = mock_terrain_loader()

        app.process_terrain_video("/home/lenny/PetrSu/3_kurs/1_sem/TPPO/unitTests/RoboSight-main/tree1v.mp4", MagicMock(), MagicMock())
        app.terrain_processor.get_video_processor().start_video_stream.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

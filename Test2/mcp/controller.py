"""Exposes Python methods to be called by the GUI."""

import os
import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from .checkbox_model import CheckBoxModel
from .image_processors import ImageProcessor


class MainController(QObject):

    updateProgress = pyqtSignal(int, str)
    changeImageSignal = pyqtSignal()
    imageChangedSignal = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.img_cv = None
        self.image_processor = ImageProcessor("","")
        self.selected_functions = set()
        dummy_data = [
            {"id": 1, "text": "Apply Median Filter", "value": 0},
            {"id": 2, "text": "Apply Scharr Filter", "value": 0},
            {"id": 3, "text": "Swap Threshold", "value": 0}
        ]
        # use this format for functions
        self.imgFilterModel = CheckBoxModel(dummy_data)

    @pyqtSlot()
    def process_image(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_processor.img_path = os.path.join(script_dir, '06.jpg')
        self.img_cv = cv2.imread(self.image_processor.img_path)
        if self.img_cv is None:
            self.updateProgress.emit(-1, "Could not read image")
            return
        self.updateProgress.emit(100, "Image added!")
        self.changeImageSignal.emit()

    @pyqtSlot(result='QString')
    def get_pixmap(self):
        unique_id = np.random.randint(1, 1000)
        return "image://imageProvider/" + str(unique_id)

    @pyqtSlot()
    def apply_filter_changes(self): #combine these with functions, store graph as different variable
        """Retrieve changes made by the user and apply to image/graph."""
        filter_applied = False

        # Original image copy
        img_bin = self.img_cv.copy()

        for val in self.imgFilterModel.list_data:
            # Median Filter
            if val["id"] == 1 and val.get("value", 0) == 1:
                filter_applied = True
                if len(img_bin.shape) == 3:  # Convert to grayscale for median if needed
                    img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
                img_bin = cv2.medianBlur(img_bin, 5)

            # Scharr Filter
            elif val["id"] == 2 and val.get("value", 0) == 1:
                filter_applied = True
                if len(img_bin.shape) == 3:
                    img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
                grad_x = cv2.Scharr(img_bin, cv2.CV_64F, 1, 0)
                grad_y = cv2.Scharr(img_bin, cv2.CV_64F, 0, 1)
                img_bin = cv2.convertScaleAbs(
                    cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
                )

            # Swap Threshold
            elif val["id"] == 3 and val.get("value", 0) == 1:
                filter_applied = True
                if len(img_bin.shape) == 3:
                    gray = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img_bin
                threshold_val, _ = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                if val.get("mask_val", 0) == 0:
                    _, img_bin = cv2.threshold(
                        gray, threshold_val, 255, cv2.THRESH_BINARY_INV
                    )
                else:
                    _, img_bin = cv2.threshold(
                        gray, threshold_val, 255, cv2.THRESH_BINARY
                    )

        self.img_cv = img_bin

        if not filter_applied:
            self.process_image()

        self.changeImageSignal.emit()

    @pyqtSlot('QString', result='QString')
    def process_name(self, name: str) -> str:
        """Process the given name and return a greeting message."""
        self.wait()
        return f"Hey {name}, your name has been processed successfully."

    @pyqtSlot(str, str)
    def apply_functions(self, image_path: str, function_name: str):
        if function_name == "binarize":
            self.image_processor.binarize(image_path)
        elif function_name == "skeletonize":
            self.image_processor.skeletonize(image_path)
        elif function_name == "extractgraph":
            self.image_processor.extractgraph(image_path)
        elif function_name == "nodes":
            self.image_processor.nodes(image_path)
        elif function_name == "edges":
            self.image_processor.edges(image_path)
        elif function_name == "en_graph":
            self.image_processor.en_graph(image_path)
        elif function_name == "graphdens":
            self.image_processor.graphdens(image_path)
        elif function_name == "avdeg":
            self.image_processor.avdeg(image_path)
        elif function_name == "deg_hm":
            self.image_processor.deg_hm(image_path)
        elif function_name == "bc_hm":
            self.image_processor.bc_hm(image_path)
        else:
            print(f"Function {function_name} is not recognized.")

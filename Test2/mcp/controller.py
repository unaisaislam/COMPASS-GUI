"""Exposes Python methods to be called by the GUI."""

import os
import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage import io, color, filters, morphology, data
from scipy.ndimage import convolve
import networkx
import csv
import pandas as pd
from skimage.morphology import skeletonize
from skimage.util import invert
from skimage.graph import route_through_array
from scipy.ndimage import label
from skimage.morphology import medial_axis
from skimage.measure import regionprops
from pathlib import Path
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
            {"id": 3, "text": "Swap Threshold", "value": 0},
            {"id":4, "text": "To PDF: Binary", "value":0},
            {"id":5, "text": "To PDF: Skeleton", "value":0},
            {"id":6, "text": "To PDF: Colored Components Graph", "value":0},
            {"id":7, "text": "To PDF: Nodes Count", "value":0},
            {"id":8, "text": "To PDF: Edge Count", "value":0},
            {"id":9, "text": "To PDF: Edge-Node Skeleton Graph", "value":0},
            {"id":10, "text": "To PDF: Graph Density", "value":0},
            {"id":11, "text": "To PDF: Average Degree", "value":0},
            {"id":12, "text": "To PDF: Degree Histogram", "value":0},
            {"id":13, "text": "To PDF: Betweenness Centrality Histogram", "value":0},
            {"id": 14, "text": "CSV: Edge Lengths and Widths", "value":0},
            {"id": 15, "text": "CSV: Node Radii", "value":0}
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
            
            # Binarize
            
            elif val["id"] == 4 and val.get("value,0") ==1:
                filter_applied = True
                self.image_processor.binarize()
            # Skeletonize
            elif val["id"] == 5 and val.get("value,0") ==1:
                filter_applied = True
                self.image_processor.skeletonize()
            # Extract Graph
            elif val["id"] == 6 and val.get("value,0") ==1:
                filter_applied = True
                self.image_processor.extractgraph()
            # Nodes
            elif val["id"] == 7 and val.get("value,0") ==1:
                filter_applied = True
                self.image_processor.nodes()    
            # Edges
            elif val["id"] == 8 and val.get("value,0") ==1:
                filter_applied = True
                self.image_processor.edges()
            # Edge-Node Graph
            elif val["id"] == 9 and val.get("value,0") ==1:
                filter_applied = True
                self.image_processor.en_graph()
            # Graph Density
            elif val["id"] == 10 and val.get("value,0") ==1:
                filter_applied = True
                self.image_processor.graphdens()
            # Average Degree
            elif val["id"] == 11 and val.get("value,0") ==1:
                filter_applied = True
                self.image_processor.avdeg()
            # Degree Heatmap
            elif val["id"] == 12 and val.get("value,0") ==1:
                filter_applied = True
                self.image_processor.deg_hm()
            # Betweenness Centrality Heatmap
            elif val["id"] == 13 and val.get("value,0") ==1:
                filter_applied = True
                self.image_processor.bc_hm()
            # Edge Lengths and Widths CSV
            elif val["id"] == 14 and val.get("value,0") ==1:
                filter_applied = True
                self.image_processor.edge_lenwid()
            # Node Radii CSV
            elif val["id"] == 15 and val.get("value,0") ==1:
                filter_applied = True
                self.image_processor.node_rad()
                

        self.img_cv = img_bin

        if not filter_applied:
            self.process_image()

        self.changeImageSignal.emit()
        
    @pyqtSlot()
    def export(self): #combine these with functions, store graph as different variable
        """Retrieve changes made by the user and apply to image/graph."""
        with PdfPages(f"{self.image_processor.image_path} figures.pdf") as pdf: 
            function_applied = False
            for val in self.imgFunctionModel.list_data:
                # Binarize
                if val["id"] == 4 and val.get("value,0") ==1:
                    function_applied = True
                    self.process_image.binarize()
                    pdf.savefig()
                    plt.close()
                # Skeletonize
                if val["id"] == 5 and val.get("value,0") ==1:
                    function_applied = True
                    self.process_image.skeletonize()
                    pdf.savefig()
                    plt.close()
                    
                # Extract Graph
                if val["id"] == 6 and val.get("value,0") ==1:
                    function_applied = True
                    self.process_image.extractgraph()
                    pdf.savefig()
                    plt.close()
                # Nodes
                if val["id"] == 7 and val.get("value,0") ==1:
                    function_applied = True
                    self.process_image.nodes()
                    pdf.savefig()
                    plt.close() 
                # Edges
                if val["id"] == 8 and val.get("value,0") ==1:
                    function_applied = True
                    self.process_image.edges()
                    pdf.savefig()
                    plt.close() 
                # Edge-Node Graph
                if val["id"] == 9 and val.get("value,0") ==1:
                    function_applied = True
                    self.process_image.en_graph()
                    pdf.savefig()
                    plt.close()
                # Graph Density
                if val["id"] == 10 and val.get("value,0") ==1:
                    function_applied = True
                    self.process_image.graphdens()
                    pdf.savefig()
                    plt.close() 
                # Average Degree
                if val["id"] == 11 and val.get("value,0") ==1:
                    function_applied = True
                    self.process_image.avdeg()
                    pdf.savefig()
                    plt.close() 
                # Degree Heatmap
                if val["id"] == 12 and val.get("value,0") ==1:
                    function_applied = True
                    self.process_image.deg_hm()
                    pdf.savefig()
                    plt.close()
                # Betweenness Centrality Heatmap
                if val["id"] == 13 and val.get("value,0") ==1:
                    function_applied = True
                    self.process_image.bc_hm()
                    pdf.savefig()
                    plt.close()
                # # Edge Lengths and Widths CSV
                # if val["id"] == 14 and val.get("value,0") ==1:
                #     function_applied = True
                #     self.image_processor.edge_lenwid()
                    # edges_df.to_csv(self.output_csv, index=False, float_format="%.2f")
                #     nodes_df.to_csv(self.output_csv, index=False, float_format="%.2f")
                # # Node Radii CSV
                # if val["id"] == 15 and val.get("value,0") ==1:
                #     function_applied = True
                #     self.image_processor.node_rad()

            if not function_applied:
                self.export()

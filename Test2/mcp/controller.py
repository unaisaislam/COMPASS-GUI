"""Exposes Python methods to be called by the GUI."""

import os
import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QVariant
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
# from .image_processors import ImageProcessor


class MainController(QObject):

    updateProgress = pyqtSignal(int, str)
    changeImageSignal = pyqtSignal()
    imageChangedSignal = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.img_cv = None
        # self.image_processor = ImageProcessor("","")
        self.selected_functions = set()
        dummy_data = [
            {"id": 1, "text": "Apply Median Filter", "value": 0},
            {"id": 2, "text": "Apply Scharr Filter", "value": 0},
            {"id": 3, "text": "Swap Threshold", "value": 0},
            {"id":4, "text": "Skeletonize", "value":0},
            {"id":5, "text": "Colored Components Graph", "value":0},
            {"id":6, "text": "Edge-Node Skeletonize", "value":0},
            {"id":7, "text": "Degree Heatmap", "value":0},
            {"id":8, "text": "Betweenness Centrality Heatmap", "value":0},
            ]
        # use this format for functions
        self.imgFilterModel = CheckBoxModel(dummy_data)

    @pyqtSlot()
    def process_image(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(script_dir, '06.jpg')
        self.img_cv = cv2.imread(img_path)
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
            # Skeletonize
            elif val["id"] == 4 and val.get("value", 0) == 1:
                filter_applied = True
                if len(img_bin.shape) == 3:
                    img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(img_bin, 127, 255, cv2.THRESH_BINARY)
                binary = binary // 255    
                skeleton = skeletonize(binary)
                img_bin = (skeleton * 255).astype("uint8")
            # Colored Components Graph
            elif val["id"] == 5 and val.get("value", 0) == 1:
                filter_applied = True
                gray = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY) if len(img_bin.shape) == 3 else img_bin
                binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                binary = binary // 255
                skeleton = skeletonize(binary)

                h, w = skeleton.shape
                graph = networkx.Graph()
                pixel_to_node = {}

                for y in range(h):
                    for x in range(w):
                        if skeleton[y, x]:
                            node_id = len(pixel_to_node)
                            pixel_to_node[(x, y)] = node_id
                            graph.add_node(node_id)

                for (x, y), node_id in pixel_to_node.items():
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx_, ny_ = x + dx, y + dy
                            if (nx_, ny_) in pixel_to_node:
                                graph.add_edge(node_id, pixel_to_node[(nx_, ny_)])

                components = list(networkx.connected_components(graph))
                component_dict = {node: i for i, comp in enumerate(components) for node in comp}
                node_colors = [component_dict[node] for node in graph.nodes()]
                positions = {node: (x * 2, y * 2) for (x, y), node in pixel_to_node.items()}

                self.graph_data = {
                    "graph": graph,
                    "positions": positions,
                    "colors": node_colors,
                    "num_components": len(components)
                }

                # Overlay graph on image
                gray = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY) if len(img_bin.shape) == 3 else img_bin
                overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel for color drawing
                # ASK FOR HELP HERE 
                img_bin = overlay
            # Edge-Node Skeletonize
            elif val["id"] == 6 and val.get("value", 0) == 1:
                filter_applied = True
                gray = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY) if len(img_bin.shape) == 3 else img_bin
                gray = gray.astype("float32") / 255.0
                threshold = filters.threshold_otsu(gray)
                binary = gray < threshold
                skel = morphology.skeletonize(binary)

                h, w = skel.shape
                pixel_graph = {}

                for y in range(h):
                    for x in range(w):
                        if not skel[y, x]:
                            continue
                        nbrs = []
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                if dy == dx == 0:
                                    continue
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                                    nbrs.append((ny, nx))
                        pixel_graph[(y, x)] = nbrs

                degree = {pix: len(nbrs) for pix, nbrs in pixel_graph.items()}
                junctions = [pix for pix, d in degree.items() if d != 2]
                if not junctions and pixel_graph:
                    junctions = [next(iter(pixel_graph))]

                node_ids = {pix: idx for idx, pix in enumerate(junctions)}
                edges = []
                visited = set()

                for seed in junctions:
                    for nbr in pixel_graph[seed]:
                        if (seed, nbr) in visited:
                            continue
                        path = [seed, nbr]
                        visited.add((seed, nbr)); visited.add((nbr, seed))
                        prev, curr = seed, nbr
                        while degree.get(curr, 0) == 2:
                            nxt = next(p for p in pixel_graph[curr] if p != prev)
                            path.append(nxt)
                            visited.add((curr, nxt)); visited.add((nxt, curr))
                            prev, curr = curr, nxt
                        source_id = node_ids[path[0]]
                        target_id = node_ids[path[-1]]
                        edges.append((source_id, target_id, path))

                self.graph_data = {
                    "skeleton": skel,
                    "junctions": junctions,
                    "edges": edges
                }

                # Overlay graph on image
                overlay = np.stack([skel * 255] * 3, axis=-1).astype("uint8")
                for _, _, path in edges:
                    ys, xs = zip(*path)
                    for i in range(len(xs) - 1):
                        pt1 = (xs[i], ys[i])
                        pt2 = (xs[i + 1], ys[i + 1])
                        cv2.line(overlay, pt1, pt2, (0, 255, 0), 1)
                for y, x in junctions:
                    cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)
                img_bin = overlay
            # Degree Heatmap
            elif val["id"] == 7 and val.get("value", 0) == 1:
                    filter_applied = True
                    # ASK
            # Betweenness Centrality Heatmap
            elif val["id"] == 8 and val.get("value", 0) == 1:
                    filter_applied = True
                    # ASK

        self.img_cv = img_bin
        if not filter_applied:
            self.process_image()

        self.changeImageSignal.emit()


    # @pyqtSlot('QString', result='QString')
    # def process_name(self, name: str) -> str:
    #     """Process the given name and return a greeting message."""
    #     self.wait()
    #     return f"Hey {name}, your name has been processed successfully."

# class MainController(QObject):

#     updateProgress = pyqtSignal(int, str)
#     changeImageSignal = pyqtSignal()
#     imageChangedSignal = pyqtSignal(bool)

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.img_cv = None
#         # self.image_processor = ImageProcessor("","")
#         self.selected_functions = set()
#         dummy_data = [
#             {"id": 1, "text": "Apply Median Filter", "value": 0},
#             {"id": 2, "text": "Apply Scharr Filter", "value": 0},
#             {"id": 3, "text": "Swap Threshold", "value": 0},
#             # {"id":4, "text": "Binarize", "value":0},
#             {"id":5, "text": "Skeletonize", "value":0},
#             {"id":6, "text": "Colored Components Graph", "value":0},
#             # {"id":7, "text": "To PDF: Nodes Count", "value":0}, # ask dickson about how to display values once the run button is clicked
#             # {"id":8, "text": "To PDF: Edge Count", "value":0},
#             {"id":9, "text": "Edge-Node Skeleton Graph", "value":0},
#             # {"id":10, "text": "To PDF: Graph Density", "value":0},
#             # {"id":11, "text": "To PDF: Average Degree", "value":0},
#             {"id":12, "text": "Degree Heatmap", "value":0},
#             {"id":13, "text": "Betweenness Centrality Heatmap", "value":0},
#             # {"id": 14, "text": "CSV: Edge Lengths and Widths", "value":0},
#             # {"id": 15, "text": "CSV: Node Radii", "value":0}
#             ]
#         # use this format for functions
#         self.imgFilterModel = CheckBoxModel(dummy_data)

#     @pyqtSlot()
#     def process_image(self):
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         self.image_processor.img_path = os.path.join(script_dir, '06.jpg')
#         self.img_cv = cv2.imread(self.image_processor.img_path)
#         if self.img_cv is None:
#             self.updateProgress.emit(-1, "Could not read image")
#             return
#         self.updateProgress.emit(100, "Image added!")
#         self.changeImageSignal.emit()

#     @pyqtSlot(result='QString')
#     def get_pixmap(self):
#         unique_id = np.random.randint(1, 1000)
#         return "image://imageProvider/" + str(unique_id)

#     @pyqtSlot()
#     def apply_filter_changes(self): 
#         """Retrieve changes made by the user and apply to image/graph."""
#         filter_applied = False

#         # Original image copy
#         img_bin = self.img_cv.copy()

#         for val in self.imgFilterModel.list_data:
#             # Median Filter
#             if val["id"] == 1 and val.get("value", 0) == 1:
#                 filter_applied = True
#                 if len(img_bin.shape) == 3:  # Convert to grayscale for median if needed
#                     img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
#                 img_bin = cv2.medianBlur(img_bin, 5)

#             # Scharr Filter
#             elif val["id"] == 2 and val.get("value", 0) == 1:
#                 filter_applied = True
#                 if len(img_bin.shape) == 3:
#                     img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
#                 grad_x = cv2.Scharr(img_bin, cv2.CV_64F, 1, 0)
#                 grad_y = cv2.Scharr(img_bin, cv2.CV_64F, 0, 1)
#                 img_bin = cv2.convertScaleAbs(
#                     cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
#                 )

#             # Swap Threshold
#             elif val["id"] == 3 and val.get("value", 0) == 1:
#                 filter_applied = True
#                 if len(img_bin.shape) == 3:
#                     gray = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
#                 else:
#                     gray = img_bin
#                 threshold_val, _ = cv2.threshold(
#                     gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
#                 )
#                 if val.get("mask_val", 0) == 0:
#                     _, img_bin = cv2.threshold(
#                         gray, threshold_val, 255, cv2.THRESH_BINARY_INV
#                     )
#                 else:
#                     _, img_bin = cv2.threshold(
#                         gray, threshold_val, 255, cv2.THRESH_BINARY
#                     )
            
#             # Skeletonize
            
#             elif val["id"] == 5 and val.get("value,0") ==1:
#                 filter_applied = True
#             if len(img_bin.shape) == 3:
#                 img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
#                 _, binary = cv2.threshold(img_bin, 127, 255, cv2.THRESH_BINARY)
#                 binary = binary // 255  # Convert to 0/1
#                 skeleton = skeletonize(binary)
#                 img_bin = (skeleton * 255).astype("uint8")

#             # # Extract Graph
#             # elif val["id"] == 6 and val.get("value,0") ==1:
#             #     filter_applied = True    
#             #     if len(img_bin.shape) == 3:
#             #         img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
#             #     _, binary = cv2.threshold(img_bin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             #     binary = binary // 255  # Normalize to 0/1

#             #     skeleton = morphology.skeletonize(binary)
#             #     h, w = skeleton.shape
#             #     graph = nx.Graph()
#             #     pixel_to_node = {}

#             #     # Add nodes
#             #     for y in range(h):
#             #         for x in range(w):
#             #             if skeleton[y, x]:
#             #                 pixel_to_node[(x, y)] = len(pixel_to_node)
#             #                 graph.add_node(pixel_to_node[(x, y)])

#             #     # Add edges
#             #     for (x, y), node_id in pixel_to_node.items():
#             #         for dx in [-1, 0, 1]:
#             #             for dy in [-1, 0, 1]:
#             #                 if dx == 0 and dy == 0:
#             #                     continue
#             #                 nx, ny = x + dx, y + dy
#             #                 if (nx, ny) in pixel_to_node:
#             #                     graph.add_edge(node_id, pixel_to_node[(nx, ny)])

#             #     # Component coloring
#             #     components = list(nx.connected_components(graph))
#             #     component_dict = {}
#             #     for i, comp in enumerate(components):
#             #         for node in comp:
#             #             component_dict[node] = i

#             #     node_colors = [component_dict[node] for node in graph.nodes()]
#             #     positions = {node: (x * 2, y * 2) for (x, y), node in pixel_to_node.items()}

#             #     # Store for visualization
#             #     self.graph_data = {
#             #         "graph": graph,
#             #         "positions": positions,
#             #         "colors": node_colors,
#             #         "num_components": len(components)
#             #     }

#             # # Nodes
#             # elif val["id"] == 7 and val.get("value,0") ==1:
#             #     filter_applied = True
#             #     self.image_processor.nodes()    
#             # # Edges
#             # elif val["id"] == 8 and val.get("value,0") ==1:
#             #     filter_applied = True
#             #     self.image_processor.edges()
#             # # Edge-Node Graph
#             # elif val["id"] == 9 and val.get("value,0") ==1:
#             #     filter_applied = True
#             #     self.image_processor.en_graph()
#             # # Graph Density
#             # elif val["id"] == 10 and val.get("value,0") ==1:
#             #     filter_applied = True
#             #     self.image_processor.graphdens()
#             # # Average Degree
#             # elif val["id"] == 11 and val.get("value,0") ==1:
#             #     filter_applied = True
#             #     self.image_processor.avdeg()
#             # # Degree Heatmap
#             # elif val["id"] == 12 and val.get("value,0") ==1:
#             #     filter_applied = True
#             #     self.image_processor.deg_hm()
#             # # Betweenness Centrality Heatmap
#             # elif val["id"] == 13 and val.get("value,0") ==1:
#             #     filter_applied = True
#             #     self.image_processor.bc_hm()
#             # # Edge Lengths and Widths CSV
#             # elif val["id"] == 14 and val.get("value,0") ==1:
#             #     filter_applied = True
#             #     # self.image_processor.edge_lenwid()
#             # # Node Radii CSV
#             # elif val["id"] == 15 and val.get("value,0") ==1:
#             #     filter_applied = True
#             #     # self.image_processor.node_rad()
                

#         self.img_cv = img_bin

#         if not filter_applied:
#             self.process_image()

#         self.changeImageSignal.emit()
    
            
#     # @pyqtSlot()
#     # def export(self): 
#     #     """Retrieve changes made by the user and apply to image/graph."""
#     #     with PdfPages(f"{self.image_processor.image_path}figures.pdf") as pdf: 
#     #         function_applied = False
#     #         for val in self.image_processor.imgFunctionModel.list_data:
#     #             # Binarize
#     #             if val["id"] == 4 and val.get("value,0") ==1:
#     #                 function_applied = True
#     #                 self.image_processor.binarize()
#     #                 pdf.savefig()
#     #                 plt.close()
#     #             # Skeletonize
#     #             if val["id"] == 5 and val.get("value,0") ==1:
#     #                 function_applied = True
#     #                 self.image_processor.skeletonize()
#     #                 pdf.savefig()
#     #                 plt.close()
                    
#     #             # Extract Graph
#     #             if val["id"] == 6 and val.get("value,0") ==1:
#     #                 function_applied = True
#     #                 self.image_processor.extractgraph()
#     #                 pdf.savefig()
#     #                 plt.close()
#     #             # Nodes
#     #             if val["id"] == 7 and val.get("value,0") ==1:
#     #                 function_applied = True
#     #                 self.image_processor.nodes()
#     #                 pdf.savefig()
#     #                 plt.close() 
#     #             # Edges
#     #             if val["id"] == 8 and val.get("value,0") ==1:
#     #                 function_applied = True
#     #                 self.image_processor.edges()
#     #                 pdf.savefig()
#     #                 plt.close() 
#     #             # Edge-Node Graph
#     #             if val["id"] == 9 and val.get("value,0") ==1:
#     #                 function_applied = True
#     #                 self.image_processor.en_graph()
#     #                 pdf.savefig()
#     #                 plt.close()
#     #             # Graph Density
#     #             if val["id"] == 10 and val.get("value,0") ==1:
#     #                 function_applied = True
#     #                 self.image_processor.graphdens()
#     #                 pdf.savefig()
#     #                 plt.close() 
#     #             # Average Degree
#     #             if val["id"] == 11 and val.get("value,0") ==1:
#     #                 function_applied = True
#     #                 self.image_processor.avdeg()
#     #                 pdf.savefig()
#     #                 plt.close() 
#     #             # Degree Heatmap
#     #             if val["id"] == 12 and val.get("value,0") ==1:
#     #                 function_applied = True
#     #                 self.image_processor.deg_hm()
#     #                 pdf.savefig()
#     #                 plt.close()
#     #             # Betweenness Centrality Heatmap
#     #             if val["id"] == 13 and val.get("value,0") ==1:
#     #                 function_applied = True
#     #                 self.image_processor.bc_hm()
#     #                 pdf.savefig()
#     #                 plt.close()
#     #     # Edge Lengths and Widths CSV
#     #     if val["id"] == 14 and val.get("value,0") ==1:
#     #         function_applied = True
#     #         self.image_processor.edge_lenwid()
#     #     # Node Radii CSV
#     #     if val["id"] == 15 and val.get("value,0") ==1:
#     #         function_applied = True
#     #         self.image_processor.node_rad()

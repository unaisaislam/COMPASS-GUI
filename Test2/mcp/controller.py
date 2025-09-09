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
import io
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
from PyQt6.QtCore import QUrl
# from .image_processors import ImageProcessor


class MainController(QObject):

    updateProgress = pyqtSignal(int, str)
    changeImageSignal = pyqtSignal()
    imageChangedSignal = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.img_cv = None
        self.img_pix = None
        # self.image_processor = ImageProcessor("","")
        self.selected_functions = set()
        dummy_data = [
            {"id": 1, "text": "Filter: Median", "value": 0},
            {"id": 2, "text": "Filter: Scharr", "value": 0},
            {"id": 3, "text": "Swap Threshold", "value": 0},
            {"id":4, "text": "Extract: Skeleton", "value":0},
            {"id":5, "text": "Extract: Colored Components Graph", "value":0},
            {"id":6, "text": "Extract: Edge-Node Skeleton", "value":0},
            {"id":7, "text": "Compute: Degree Heatmap", "value":0},
            {"id":8, "text": "Compute: Betweenness Centrality Heatmap", "value":0},
            ]
        # use this format for functions
        self.imgFilterModel = CheckBoxModel(dummy_data)

    def plot_to_opencv(self,fig):
        """Convert a Matplotlib figure to an OpenCV image."""
        if fig:
            # Save a figure to a buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)

            # Convert buffer to NumPy array
            img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()

            # Decode image including the alpha channel (if any)
            img_cv_rgba = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

            # Convert RGBA to RGB if needed
            if img_cv_rgba.shape[2] == 4:
                img_cv_rgb = cv2.cvtColor(img_cv_rgba, cv2.COLOR_RGBA2RGB)
            else:
                img_cv_rgb = img_cv_rgba

            # Convert RGB to BGR to match OpenCV color space
            img_cv_bgr = cv2.cvtColor(img_cv_rgb, cv2.COLOR_RGB2BGR)
            return img_cv_bgr
        return None
    # use this then update self.image_cv 

    @pyqtSlot(str)
    def process_image(self, img_path):
        # Convert file URL (file:///C:/...) to local file path (C:\...)
        local_path = QUrl(img_path).toLocalFile()

        self.img_cv = cv2.imread(local_path)
        if self.img_cv is None:
            self.updateProgress.emit(-1, "Could not read image")
            return

        self.img_pix = self.img_cv.copy()
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
            if val.get("value", 0) == 1:
                filter_applied = True
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
                adj_list = {}

                graph = networkx.Graph()
                graph.add_nodes_from(adj_list.keys())
                
                            
                def add_edge(g, a, b):
                    g.setdefault(a,[]).append(b)
                    g.setdefault(b,[]).append(a)
                    
                for y in range(h):
                    for x in range(w):
                        if not skeleton[y, x]:
                            continue
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dy == 0 and dx == 0:
                                    continue
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                                    add_edge(adj_list, (y, x), (ny, nx))
                                    
                for u, neighbors in adj_list.items():
                    for v in neighbors:
                        if not graph.has_edge(u, v):
                            graph.add_edge(u, v)

                components = list(networkx.connected_components(graph))
                number_of_components = len(components)

                component_index = {node: i for i, component in enumerate(components) for node in component}
                color_list = [component_index[node] for node in graph.nodes()]
                position = {(y,x): (x, -y) for y, x in graph.nodes()}
                
                fig, ax = plt.subplots(figsize=(8,8), dpi=100)
                networkx.draw(
                    graph,
                    pos = position,
                    node_color=color_list,
                    node_size=5,
                    cmap=plt.cm.get_cmap('magma', number_of_components),
                    with_labels=False,
                    edge_color='lightgray',
                    ax=ax
                )
                ax.axis('off')
                img_bin = self.plot_to_opencv(fig)  
                plt.close(fig)

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
                    gray = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY) if len(img_bin.shape) == 3 else img_bin
                    gray = gray.astype("float32") / 255.0
                    threshold = filters.threshold_otsu(gray)
                    binary = gray < threshold
                    skeleton = morphology.skeletonize(binary)
                    h, w = skeleton.shape
                    adj_list = {}
                    graph = networkx.Graph()
                    graph.add_nodes_from(adj_list.keys())        
                    def add_edge(g, a, b):
                        g.setdefault(a,[]).append(b)
                        g.setdefault(b,[]).append(a)
                        
                    for y in range(h):
                        for x in range(w):
                            if not skeleton[y, x]:
                                continue
                            for dy in [-1, 0, 1]:
                                for dx in [-1, 0, 1]:
                                    if dy == 0 and dx == 0:
                                        continue
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                                        add_edge(adj_list, (y, x), (ny, nx))
                                        
                    for u, neighbors in adj_list.items():
                        for v in neighbors:
                            if not graph.has_edge(u, v):
                                graph.add_edge(u, v)
                                
                    degmap = np.zeros(skeleton.shape,dtype=float)
                    for (y,x), deg in graph.degree():
                        degmap[y,x] = deg
                        
                    fig, ax = plt.subplots(dpi=100)
                    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='magma'), ax=ax, orientation='vertical')
                    cbar.set_label('Pixel Degree', rotation=270, labelpad=15)
                    ax.imshow(gray, cmap='gray')
                    ax.imshow(degmap, cmap='magma', alpha=0.5)
                    ax.axis('off')
                    plt.tight_layout()
                    img_bin = self.plot_to_opencv(fig)  
                    plt.close(fig)
            # Betweenness Centrality Heatmap
            elif val["id"] == 8 and val.get("value", 0) == 1:
                    filter_applied = True
                    gray = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY) if len(img_bin.shape) == 3 else img_bin
                    gray = gray.astype("float32") / 255.0
                    threshold = filters.threshold_otsu(gray)
                    binary = gray < threshold
                    skeleton = morphology.skeletonize(binary)
                    h, w = skeleton.shape
                    adj_list = {}
                        
                    def add_edge(g, a, b):
                        g.setdefault(a,[]).append(b)
                        g.setdefault(b,[]).append(a)
                        
                    for y in range(h):
                        for x in range(w):
                            if not skeleton[y, x]:
                                continue
                            for dy in [-1, 0, 1]:
                                for dx in [-1, 0, 1]:
                                    if dy == 0 and dx == 0:
                                        continue
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                                        add_edge(adj_list, (y, x), (ny, nx))
                                        
                    graph = networkx.Graph()
                    graph.add_nodes_from(adj_list.keys())
                    
                    for u, neighbors in adj_list.items():
                        for v in neighbors:
                            if not graph.has_edge(u, v):
                                graph.add_edge(u, v)
                                
                    bc_dict = networkx.betweenness_centrality(graph)
                    bc_map = np.zeros(skeleton.shape, dtype=float)
                    
                    for (y,x), score in bc_dict.items():
                        bc_map[y,x] = score
                        
                    fig, ax = plt.subplots(dpi=100)
                    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='magma'), ax=ax, orientation='vertical')
                    cbar.set_label('Betweenness Centrality', rotation=270, labelpad=15)
                    ax.imshow(gray, cmap='gray')
                    ax.imshow(bc_map, cmap='magma', alpha=0.5)
                    ax.axis('off')
                    plt.tight_layout()
                    img_bin = self.plot_to_opencv(fig)  
                    plt.close(fig)
                        
        self.img_pix = img_bin

        if not filter_applied:
            self.changeImageSignal.emit()
            self.updateProgress.emit(100, "No filter applied, showing original")
            return

        self.changeImageSignal.emit()
        self.updateProgress.emit(100, "Filter applied")

        if self.img_pix is None:
            self.updateProgress.emit(-1, "No image loaded to apply filter")
            return

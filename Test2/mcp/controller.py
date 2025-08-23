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
                    img = io.imread(self.image_path)
                    gray = color.rgb2gray(img)
                    threshold = filters.threshold_otsu(gray)
                    binary = gray < threshold
                    plt.imshow(binary, cmap='gray')
                    plt.axis('off')
                    plt.title('Binary Image')
                    pdf.savefig()
                    plt.close()
                # Skeletonize
                if val["id"] == 5 and val.get("value,0") ==1:
                    function_applied = True
                    img = io.imread(self.image_path)
                    gray = color.rgb2gray(img)
                    threshold = filters.threshold_otsu(gray)
                    binary = gray < threshold
                    skeleton = morphology.skeletonize(binary)
                    plt.imshow(skeleton, cmap='gray')
                    plt.axis('off')
                    plt.title('Skeletonized Image')
                    pdf.savefig()
                    plt.close()
                    
                # Extract Graph
                if val["id"] == 6 and val.get("value,0") ==1:
                    function_applied = True
                    img = io.imread(self.image_path)
                    gray = color.rgb2gray(img)
                    threshold = filters.threshold_otsu(gray)
                    binary = gray < threshold
                    skel = morphology.skeletonize(binary)
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
                    plt.figure(figsize=(5, 5))
                    networkx.draw(
                        graph,
                        pos = position,
                        node_color=color_list,
                        node_size=5,
                        cmap=plt.cm.get_cmap('magma', number_of_components),
                        with_labels=False,
                        edge_color='lightgray',
                    )
                    plt.title('Colored Component Graph')
                    plt.axis('off')
                    pdf.savefig()
                    plt.close()
                # Nodes
                if val["id"] == 7 and val.get("value,0") ==1:
                    function_applied = True
                    img = io.imread(self.image_path)
                    gray = color.rgb2gray(img)
                    threshold = filters.threshold_otsu(gray)
                    binary = gray < threshold
                    skeleton = morphology.skeletonize(binary)
                    kernel = np.array([[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]])
                    neighbor_counts = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
                    nodes = np.logical_and(skeleton, neighbor_counts == 1)
                    fig = plt.figure()
                    plt.title('Nodes')
                    plt.axis('off')
                    plt.text(0.5, 0.5, str(nodes), ha='center', va='center', size=24) 
                    pdf.savefig(fig)
                    plt.close() 
                # Edges
                if val["id"] == 8 and val.get("value,0") ==1:
                    function_applied = True
                    img = io.imread(self.image_path)
                    gray = color.rgb2gray(img)
                    threshold = filters.threshold_otsu(gray)
                    binary = gray < threshold
                    skeleton = morphology.skeletonize(binary)
                    kernel = np.array([[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]])
                    neighbor_counts = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
                    edges = np.logical_and(skeleton, neighbor_counts >= 3)
                    fig = plt.figure()
                    plt.title('Edges')
                    plt.axis('off')
                    plt.text(0.5, 0.5, str(edges), ha='center', va='center', size=24) 
                    pdf.savefig(fig)
                    plt.close() 
                # Edge-Node Graph
                if val["id"] == 9 and val.get("value,0") ==1:
                    function_applied = True
                    image = Path(self.image_path)
                    img = io.imread(str(image))
                    gray = color.rgb2gray(img)
                    threshold = filters.threshold_otsu(gray)
                    binary = gray < threshold
                    skel = morphology.skeletonize(binary)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(skel, cmap='gray')
                    ax.set_title("Skeleton with Nodes (green) and Edges (red)")
                    ax.axis('off')
                    pixel_graph = {}
                    h, w = skel.shape
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
                            while degree[curr] == 2:
                                nxt = next(p for p in pixel_graph[curr] if p != prev)
                                path.append(nxt)
                                visited.add((curr, nxt)); visited.add((nxt, curr))
                                prev, curr = curr, nxt
                            source_id = node_ids[path[0]]
                            target_id = node_ids[path[-1]]
                            edges.append((source_id, target_id, path))
                    for idx, (src, tgt, path) in enumerate(edges):
                        # Extract x/y coordinates
                        ys, xs = zip(*path)
                        ax.plot(xs, ys, color='green', linewidth=1)
                        # Label at midpoint
                        mid = len(path) // 2
                        y_mid, x_mid = path[mid]
                        ax.text(x_mid, y_mid, str(idx), color='blue', fontsize=5, ha='center', va='center')
                    for pix in junctions:
                        y, x = pix
                        ax.scatter(x, y, color='red', s=20)
                    plt.tight_layout()
                    plt.title('Edge-Node Skeleton Graph')
                    pdf.savefig()
                    plt.close()
                # Graph Density
                if val["id"] == 10 and val.get("value,0") ==1:
                    function_applied = True
                    img = io.imread(self.image_path)
                    gray = color.rgb2gray(img)
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

                    graph_density = networkx.density(graph)
                    fig = plt.figure()
                    plt.title('Graph Density')
                    plt.axis('off')
                    plt.text(0.5, 0.5, str(graph_density), ha='center', va='center', size=24) 
                    pdf.savefig(fig)
                    plt.close() 
                # Average Degree
                if val["id"] == 11 and val.get("value,0") ==1:
                    function_applied = True
                    img = io.imread(self.image_path)
                    gray = color.rgb2gray(img)
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
                                
                    deglist = [d for _, d in graph.degree()]
                    avdeg = sum(deglist)/len(deglist) if deglist else 0
                    fig = plt.figure()
                    plt.title('Average Degree')
                    plt.axis('off')
                    plt.text(0.5, 0.5, str(avdeg), ha='center', va='center', size=24) 
                    pdf.savefig(fig)
                    plt.close() 
                # Degree Heatmap
                if val["id"] == 12 and val.get("value,0") ==1:
                    function_applied = True
                    img = io.imread(self.image_path)
                    gray = color.rgb2gray(img)
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
                    plt.figure(figsize=(5,5))
                    plt.imshow(gray,cmap='gray')
                    plt.imshow(degmap, cmap='magma', alpha=0.5)
                    plt.colorbar(label='Pixel Degree')
                    plt.title('Degree Heatmap')
                    plt.axis('off')
                    pdf.savefig()
                    plt.close()
                # Betweenness Centrality Heatmap
                if val["id"] == 13 and val.get("value,0") ==1:
                    function_applied = True
                    img = io.imread(self.image_path)
                    gray = color.rgb2gray(img)
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
                        
                    plt.figure(figsize=(5,5))
                    plt.imshow(gray, cmap='gray')
                    plt.imshow(bc_map, cmap='magma', alpha=0.7)
                    plt.colorbar(label='Betweenness Centrality')
                    plt.title('Betweenness Centrality Heatmap')
                    plt.axis('off')
                    pdf.savefig()
                    plt.close()
                # # Edge Lengths and Widths CSV
                # if val["id"] == 14 and val.get("value,0") ==1:
                #     function_applied = True
                #     self.image_processor.edge_lenwid()
                # # Node Radii CSV
                # if val["id"] == 15 and val.get("value,0") ==1:
                #     function_applied = True
                #     self.image_processor.node_rad()

            if not function_applied:
                self.export()

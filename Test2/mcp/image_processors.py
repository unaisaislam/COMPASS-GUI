import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage import io, color, filters, morphology, data
from scipy.ndimage import convolve
import networkx
import cv2
import csv
import pandas as pd
from skimage.morphology import skeletonize
from skimage.util import invert
from skimage.graph import route_through_array
from scipy.ndimage import label
from skimage.morphology import medial_axis
from skimage.measure import regionprops
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSlot
from skimage import io, color, filters
import matplotlib.pyplot as plt
from .checkbox_model import CheckBoxModel

class ImageProcessor(QObject):
    def __init__(self, image_path, output_csv):
        super().__init__()
        self.image_path = image_path
        self.output_csv = output_csv
        temp_data = [
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
        self.imgFunctionModel = CheckBoxModel(temp_data)

    @pyqtSlot()
    def binarize(self):
        img = io.imread(self.image_path)
        gray = color.rgb2gray(img)
        threshold = filters.threshold_otsu(gray)
        binary = gray < threshold
        plt.imshow(binary, cmap="gray")
        plt.axis("off")
        plt.title("Binary Image")
        # plt.show()
        
    @pyqtSlot()
    def skeletonize(self):
        '''
        Binarizes the image, skeletonizes the image, and extracts Skeletonized Graph.
        '''
        img = io.imread(self.image_path)
        gray = color.rgb2gray(img)
        threshold = filters.threshold_otsu(gray)
        binary = gray < threshold
        skeleton = morphology.skeletonize(binary)
        plt.imshow(skeleton, cmap='gray')
        plt.axis('off')
        plt.title('Skeletonized Image')
        # plt.show()

    @pyqtSlot()
    def extractgraph(self):
        '''
        Binarizes the image depending on the mask value, skeletonizes the image, and extracts a Colored-Component Skeleton Graph.
        '''
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
        # plt.show()

    # number of nodes
    @pyqtSlot()
    def nodes(self):
        '''
        Returns the number of nodes in a skeletonized image.
        '''
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
        plt.figure()
        plt.title('Nodes')
        plt.axis('off')
        plt.text(0.5, 0.5, str(np.sum(nodes)), ha='center', va='center', size=24) 
        # print(f"Endpoints: {np.sum(nodes)}")
        
    # number of edges
    @pyqtSlot()
    def edges(self):
        '''
        Returns the number of edges in a skeletonized image.
        '''
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
        plt.figure()
        plt.title('Nodes')
        plt.axis('off')
        plt.text(0.5, 0.5, str(np.sum(edges)), ha='center', va='center', size=24) 
        # print(f"Endpoints: {np.sum(edges)}")


    #edge-node graph
    @pyqtSlot()
    def en_graph(self):
        '''
        Exports a skeletonized graph with colored nodes and edges.
        '''
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
        plt.axis('off')
        # plt.show()
        
    # graph density
    @pyqtSlot()
    def graphdens(self):
        '''
        Returns density of skeletonized graph of an image. Values closer to 0 mean the graph is less dense and there are less connections within the nodes relative to the total possible number of edges. Values closer to 1 mean the graph is more dense and there are more connections within the nodes relative to the total possible number of edges.
        '''
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
        
        plt.figure()
        plt.title('Graph Density')
        plt.axis('off')
        plt.text(0.5, 0.5, str(graph_density), ha='center', va='center', size=24) 
        # print(f"Graph Density: {graph_density:.4f}")
        
    # average degree
    @pyqtSlot()
    def avdeg(self):
        '''
        Returns density of skeletonized graph of an image. Values closer to 0 mean the graph is less dense and there are less connections within the nodes relative to the total possible number of edges. Values closer to 1 mean the graph is more dense and there are more connections within the nodes relative to the total possible number of edges.
        '''
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
        
        plt.figure()
        plt.title('Average Degree')
        plt.axis('off')
        plt.text(0.5, 0.5, str(avdeg), ha='center', va='center', size=24) 
        # print(f"Average Degree: {avdeg:.4f}")
        
    # degree heatmap
    @pyqtSlot()
    def deg_hm(self):
        '''
        Exports a heatmap of the degrees across the graph's skeleton based on an image.
        '''
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
        # plt.show()
        
    # betweeness centrality heatmap
    @pyqtSlot()
    def bc_hm(self):
        '''
        Exports a heatmap of the betweeness-centrality across the graph's skeleton based on an image.
        '''
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
        # plt.show()

    # edge length and width csv
    @pyqtSlot()
    def edge_lenwid(self):
        '''
        Finds the length and width of all edges in graph and exports to CSV.        '''
        def march_until_background(mask: np.ndarray, start: np.ndarray, direction: np.ndarray) -> int:
            """
            Step from `start` in unit-vector `direction` until mask==False or out of bounds.
            Returns number of foreground steps.
            """
            steps = 0
            h, w = mask.shape
            y, x = float(start[0]), float(start[1])
            while True:
                y += direction[0]
                x += direction[1]
                yi, xi = int(round(y)), int(round(x))
                if not (0 <= yi < h and 0 <= xi < w and mask[yi, xi]):
                    break
                steps += 1
            return steps
        image = Path(self.image_path)
        img = io.imread(image)
        gray = color.rgb2gray(img)
        threshold = filters.threshold_otsu(gray)
        binary = gray < threshold
        skel = morphology.skeletonize(binary)
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

        # node pixels and map to IDs
        degree = {pix: len(nbrs) for pix, nbrs in pixel_graph.items()}
        junctions = [pix for pix, d in degree.items() if d != 2]
        if not junctions and pixel_graph:
            junctions = [next(iter(pixel_graph))]
        # node ID mapping
        node_ids = {pix: idx for idx, pix in enumerate(junctions)}
        visited = set()
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

        records = []
        for source_id, target_id, path in edges:
            # Length along path
            length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
                        for i in range(len(path)-1))
            # Compute perpendicular at midpoint
            mid = len(path)//2
            i0 = mid-1 if mid>0 else mid
            i1 = mid+1 if mid<len(path)-1 else mid
            tangent = np.array(path[i1]) - np.array(path[i0])
            perp = np.array([-tangent[1], tangent[0]])
            if np.linalg.norm(perp) != 0:
                perp = perp / np.linalg.norm(perp)
            center = np.array(path[mid], float)
            w1 = march_until_background(binary, center,  perp)
            w2 = march_until_background(binary, center, -perp)
            width = w1 + w2
            records.append({
                "Source": source_id,
                "Target": target_id,
                "Weight": width,
                "Length": length
            })

        edges_df = pd.DataFrame(records)
        # edges_df.to_csv(self.output_csv, index=False, float_format="%.2f")
        # print(f"Saved edge list to {self.output_csv}.")
        

    # # export to pdf
    # @pyqtSlot()
    # def allpdfs(self):
    #     '''
    #     Exports Binary, Skeleton, Skeleton with Nodes/Edges, Colored Components, Degree Heatmap, and Betweenness-Centrality Heatmap all into 1 PDF.
    #     Mask value must be 0 (binary < threshold) or 1 (binary > threshold)
    #     '''
    #     with PdfPages(f"{self.image_path} figures.pdf") as pdf:
    #         # binary
    #         img = io.imread(self.image_path) #instead of img_path
    #         gray = color.rgb2gray(img)
    #         threshold = filters.threshold_otsu(gray)
    #         binary = gray < threshold
    #         plt.imshow(binary, cmap='gray')
    #         plt.axis('off')
    #         plt.title('Binary Image')
    #         pdf.savefig()
    #         plt.close()

    #         # skeleton
    #         skeleton = morphology.skeletonize(binary)
    #         plt.imshow(skeleton, cmap='gray')
    #         plt.axis('off')
    #         plt.title('Skeletonized Image')
    #         pdf.savefig()
    #         plt.close()
            
    #         #skeleton with edges
    #         image = Path(self.image_path)
    #         img = io.imread(str(image))
    #         gray = color.rgb2gray(img)
    #         threshold = filters.threshold_otsu(gray)
    #         binary = gray < threshold
    #         skel = morphology.skeletonize(binary)
    #         fig, ax = plt.subplots(figsize=(8, 8))
    #         ax.imshow(skel, cmap='gray')
    #         ax.set_title("Skeleton with Nodes (green) and Edges (red)")
    #         ax.axis('off')
    #         pixel_graph = {}
    #         h, w = skel.shape
    #         for y in range(h):
    #             for x in range(w):
    #                 if not skel[y, x]:
    #                     continue
    #                 nbrs = []
    #                 for dy in (-1, 0, 1):
    #                     for dx in (-1, 0, 1):
    #                         if dy == dx == 0:
    #                             continue
    #                         ny, nx = y + dy, x + dx
    #                         if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
    #                             nbrs.append((ny, nx))
    #                 pixel_graph[(y, x)] = nbrs
    #         degree = {pix: len(nbrs) for pix, nbrs in pixel_graph.items()}
    #         junctions = [pix for pix, d in degree.items() if d != 2]
    #         if not junctions and pixel_graph:
    #             junctions = [next(iter(pixel_graph))]
    #         node_ids = {pix: idx for idx, pix in enumerate(junctions)}
    #         edges = []
    #         visited = set()
    #         for seed in junctions:
    #             for nbr in pixel_graph[seed]:
    #                 if (seed, nbr) in visited:
    #                     continue
    #                 path = [seed, nbr]
    #                 visited.add((seed, nbr)); visited.add((nbr, seed))
    #                 prev, curr = seed, nbr
    #                 while degree[curr] == 2:
    #                     nxt = next(p for p in pixel_graph[curr] if p != prev)
    #                     path.append(nxt)
    #                     visited.add((curr, nxt)); visited.add((nxt, curr))
    #                     prev, curr = curr, nxt
    #                 source_id = node_ids[path[0]]
    #                 target_id = node_ids[path[-1]]
    #                 edges.append((source_id, target_id, path))
    #         for idx, (src, tgt, path) in enumerate(edges):
    #             # Extract x/y coordinates
    #             ys, xs = zip(*path)
    #             ax.plot(xs, ys, color='green', linewidth=1)
    #             # Label at midpoint
    #             mid = len(path) // 2
    #             y_mid, x_mid = path[mid]
    #             ax.text(x_mid, y_mid, str(idx), color='blue', fontsize=5, ha='center', va='center')
    #         for pix in junctions:
    #             y, x = pix
    #             ax.scatter(x, y, color='red', s=20)

    #         plt.tight_layout()
    #         pdf.savefig()
    #         plt.close()
                
    #         # colored graph
    #         img = io.imread(self.image_path)
    #         gray = color.rgb2gray(img)
    #         threshold = filters.threshold_otsu(gray)
    #         binary = gray < threshold
    #         skel = morphology.skeletonize(binary)
    #         h, w = skeleton.shape
    #         adj_list = {}

    #         graph = networkx.Graph()
    #         graph.add_nodes_from(adj_list.keys())
            
                        
    #         def add_edge(g, a, b):
    #             g.setdefault(a,[]).append(b)
    #             g.setdefault(b,[]).append(a)
                
    #         for y in range(h):
    #             for x in range(w):
    #                 if not skeleton[y, x]:
    #                     continue
    #                 for dy in [-1, 0, 1]:
    #                     for dx in [-1, 0, 1]:
    #                         if dy == 0 and dx == 0:
    #                             continue
    #                         ny, nx = y + dy, x + dx
    #                         if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
    #                             add_edge(adj_list, (y, x), (ny, nx))
                                
    #         for u, neighbors in adj_list.items():
    #             for v in neighbors:
    #                 if not graph.has_edge(u, v):
    #                     graph.add_edge(u, v)

    #         components = list(networkx.connected_components(graph))
    #         number_of_components = len(components)

    #         component_index = {node: i for i, component in enumerate(components) for node in component}
    #         color_list = [component_index[node] for node in graph.nodes()]
    #         position = {(y,x): (x, -y) for y, x in graph.nodes()}
    #         plt.figure(figsize=(5, 5))
    #         networkx.draw(
    #             graph,
    #             pos = position,
    #             node_color=color_list,
    #             node_size=5,
    #             cmap=plt.cm.get_cmap('magma', number_of_components),
    #             with_labels=False,
    #             edge_color='lightgray',
    #         )
    #         plt.title('Colored Component Graph')
    #         plt.axis('off')
    #         pdf.savefig()
    #         plt.close()
            
    #         # degree heatmap
    #         degmap = np.zeros(skeleton.shape,dtype=float)
    #         for (y,x), deg in graph.degree():
    #             degmap[y,x] = deg
    #         plt.figure(figsize=(5,5))
    #         plt.imshow(gray,cmap='gray')
    #         plt.imshow(degmap, cmap='magma', alpha=0.5)
    #         plt.colorbar(label='Pixel Degree')
    #         plt.title('Degree Heatmap')
    #         plt.axis('off')
    #         pdf.savefig()
    #         plt.close()
            
    #         # betweenness centrality heatmap
    #         bc_dict = networkx.betweenness_centrality(graph)
    #         bc_map = np.zeros(skeleton.shape, dtype=float)
            
    #         for (y,x), score in bc_dict.items():
    #             bc_map[y,x] = score
                
    #         plt.figure(figsize=(5,5))
    #         plt.imshow(gray, cmap='gray')
    #         plt.imshow(bc_map, cmap='magma', alpha=0.7)
    #         plt.colorbar(label='Betweenness Centrality')
    #         plt.title('Betweenness Centrality Heatmap on Skeleton')
    #         plt.axis('off')
    #         pdf.savefig()
    #         plt.close()

    # node radii
    @pyqtSlot()
    def node_rad(self):
        '''
        Takes the min, max, and average radii around a node (distance from the node to its boundaries) and exports into CSV along with node coordinates.
        '''
        def march_until_background(mask: np.ndarray, start: np.ndarray, direction: np.ndarray) -> int:
            """Step from `start` in unit-vector `direction` until mask==False or out of bounds."""
            steps = 0
            h, w = mask.shape
            y, x = float(start[0]), float(start[1])
            while True:
                y += direction[0]
                x += direction[1]
                yi, xi = int(round(y)), int(round(x))
                if not (0 <= yi < h and 0 <= xi < w and mask[yi, xi]):
                    break
                steps += 1
            return steps

        image = Path(self.image_path)
        img = io.imread(image)
        gray = color.rgb2gray(img)
        threshold = filters.threshold_otsu(gray)
        binary = gray < threshold
        skel = morphology.skeletonize(binary)
        pixel_graph = {}
        h, w = skel.shape
        for y in range(h):
            for x in range(w):
                if not skel[y, x]:
                    continue
                nbrs = [(y+dy, x+dx) for dy in (-1,0,1) for dx in (-1,0,1)
                        if (dy != 0 or dx != 0) and 0 <= y+dy < h and 0 <= x+dx < w and skel[y+dy, x+dx]]
                pixel_graph[(y, x)] = nbrs

        degree = {pix: len(nbrs) for pix, nbrs in pixel_graph.items()}
        junctions = [pix for pix, d in degree.items() if d != 2]
        if not junctions and pixel_graph:
            junctions = [next(iter(pixel_graph))]

        node_ids = {pix: idx for idx, pix in enumerate(junctions)}
        records = []

        for pix, node_id in node_ids.items():
            center = np.array(pix, float)
            radii = []
            for angle in np.linspace(0, 2*np.pi, num=16, endpoint=False):
                direction = np.array([np.sin(angle), np.cos(angle)])
                r = march_until_background(binary, center, direction)
                radii.append(r)
            avg_radius = np.mean(radii)
            max_radius = np.max(radii)
            min_radius = np.min(radii)
            records.append({
                "Node": node_id,
                "Y": pix[0],
                "X": pix[1],
                "AvgRadius": avg_radius,
                "MaxRadius": max_radius,
                "MinRadius": min_radius
            })

        nodes_df = pd.DataFrame(records)
        # nodes_df.to_csv(self.output_csv, index=False, float_format="%.2f")
        # print(f"Saved node radius map to {self.output_csv}.")
        
        
    # @pyqtSlot()
    # def export(self): #combine these with functions, store graph as different variable
    #     """Retrieve changes made by the user and apply to image/graph."""
    #     with PdfPages(f"{self.image_path} figures.pdf") as pdf: 
    #         function_applied = False
    #         for val in self.imgFunctionModel.list_data:
    #             # Binarize
    #             if val["id"] == 4 and val.get("value,0") ==1:
    #                 function_applied = True
    #                 img = io.imread(self.image_path)
    #                 gray = color.rgb2gray(img)
    #                 threshold = filters.threshold_otsu(gray)
    #                 binary = gray < threshold
    #                 plt.imshow(binary, cmap='gray')
    #                 plt.axis('off')
    #                 plt.title('Binary Image')
    #                 pdf.savefig()
    #                 plt.close()
    #             # Skeletonize
    #             if val["id"] == 5 and val.get("value,0") ==1:
    #                 function_applied = True
    #                 img = io.imread(self.image_path)
    #                 gray = color.rgb2gray(img)
    #                 threshold = filters.threshold_otsu(gray)
    #                 binary = gray < threshold
    #                 skeleton = morphology.skeletonize(binary)
    #                 plt.imshow(skeleton, cmap='gray')
    #                 plt.axis('off')
    #                 plt.title('Skeletonized Image')
    #                 pdf.savefig()
    #                 plt.close()
                    
    #             # Extract Graph
    #             if val["id"] == 6 and val.get("value,0") ==1:
    #                 function_applied = True
    #                 img = io.imread(self.image_path)
    #                 gray = color.rgb2gray(img)
    #                 threshold = filters.threshold_otsu(gray)
    #                 binary = gray < threshold
    #                 skel = morphology.skeletonize(binary)
    #                 h, w = skeleton.shape
    #                 adj_list = {}

    #                 graph = networkx.Graph()
    #                 graph.add_nodes_from(adj_list.keys())
                    
                                
    #                 def add_edge(g, a, b):
    #                     g.setdefault(a,[]).append(b)
    #                     g.setdefault(b,[]).append(a)
                        
    #                 for y in range(h):
    #                     for x in range(w):
    #                         if not skeleton[y, x]:
    #                             continue
    #                         for dy in [-1, 0, 1]:
    #                             for dx in [-1, 0, 1]:
    #                                 if dy == 0 and dx == 0:
    #                                     continue
    #                                 ny, nx = y + dy, x + dx
    #                                 if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
    #                                     add_edge(adj_list, (y, x), (ny, nx))
                                        
    #                 for u, neighbors in adj_list.items():
    #                     for v in neighbors:
    #                         if not graph.has_edge(u, v):
    #                             graph.add_edge(u, v)

    #                 components = list(networkx.connected_components(graph))
    #                 number_of_components = len(components)

    #                 component_index = {node: i for i, component in enumerate(components) for node in component}
    #                 color_list = [component_index[node] for node in graph.nodes()]
    #                 position = {(y,x): (x, -y) for y, x in graph.nodes()}
    #                 plt.figure(figsize=(5, 5))
    #                 networkx.draw(
    #                     graph,
    #                     pos = position,
    #                     node_color=color_list,
    #                     node_size=5,
    #                     cmap=plt.cm.get_cmap('magma', number_of_components),
    #                     with_labels=False,
    #                     edge_color='lightgray',
    #                 )
    #                 plt.title('Colored Component Graph')
    #                 plt.axis('off')
    #                 pdf.savefig()
    #                 plt.close()
    #             # Nodes
    #             if val["id"] == 7 and val.get("value,0") ==1:
    #                 function_applied = True
    #                 img = io.imread(self.image_path)
    #                 gray = color.rgb2gray(img)
    #                 threshold = filters.threshold_otsu(gray)
    #                 binary = gray < threshold
    #                 skeleton = morphology.skeletonize(binary)
    #                 kernel = np.array([[1, 1, 1],
    #                                 [1, 0, 1],
    #                                 [1, 1, 1]])
    #                 neighbor_counts = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    #                 nodes = np.logical_and(skeleton, neighbor_counts == 1)
    #                 fig = plt.figure()
    #                 plt.title('Nodes')
    #                 plt.axis('off')
    #                 plt.text(0.5, 0.5, str(nodes), ha='center', va='center', size=24) 
    #                 pdf.savefig(fig)
    #                 plt.close() 
    #             # Edges
    #             if val["id"] == 8 and val.get("value,0") ==1:
    #                 function_applied = True
    #                 img = io.imread(self.image_path)
    #                 gray = color.rgb2gray(img)
    #                 threshold = filters.threshold_otsu(gray)
    #                 binary = gray < threshold
    #                 skeleton = morphology.skeletonize(binary)
    #                 kernel = np.array([[1, 1, 1],
    #                                 [1, 0, 1],
    #                                 [1, 1, 1]])
    #                 neighbor_counts = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    #                 edges = np.logical_and(skeleton, neighbor_counts >= 3)
    #                 fig = plt.figure()
    #                 plt.title('Edges')
    #                 plt.axis('off')
    #                 plt.text(0.5, 0.5, str(edges), ha='center', va='center', size=24) 
    #                 pdf.savefig(fig)
    #                 plt.close() 
    #             # Edge-Node Graph
    #             if val["id"] == 9 and val.get("value,0") ==1:
    #                 function_applied = True
    #                 image = Path(self.image_path)
    #                 img = io.imread(str(image))
    #                 gray = color.rgb2gray(img)
    #                 threshold = filters.threshold_otsu(gray)
    #                 binary = gray < threshold
    #                 skel = morphology.skeletonize(binary)
    #                 fig, ax = plt.subplots(figsize=(8, 8))
    #                 ax.imshow(skel, cmap='gray')
    #                 ax.set_title("Skeleton with Nodes (green) and Edges (red)")
    #                 ax.axis('off')
    #                 pixel_graph = {}
    #                 h, w = skel.shape
    #                 for y in range(h):
    #                     for x in range(w):
    #                         if not skel[y, x]:
    #                             continue
    #                         nbrs = []
    #                         for dy in (-1, 0, 1):
    #                             for dx in (-1, 0, 1):
    #                                 if dy == dx == 0:
    #                                     continue
    #                                 ny, nx = y + dy, x + dx
    #                                 if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
    #                                     nbrs.append((ny, nx))
    #                         pixel_graph[(y, x)] = nbrs
    #                 degree = {pix: len(nbrs) for pix, nbrs in pixel_graph.items()}
    #                 junctions = [pix for pix, d in degree.items() if d != 2]
    #                 if not junctions and pixel_graph:
    #                     junctions = [next(iter(pixel_graph))]
    #                 node_ids = {pix: idx for idx, pix in enumerate(junctions)}
    #                 edges = []
    #                 visited = set()
    #                 for seed in junctions:
    #                     for nbr in pixel_graph[seed]:
    #                         if (seed, nbr) in visited:
    #                             continue
    #                         path = [seed, nbr]
    #                         visited.add((seed, nbr)); visited.add((nbr, seed))
    #                         prev, curr = seed, nbr
    #                         while degree[curr] == 2:
    #                             nxt = next(p for p in pixel_graph[curr] if p != prev)
    #                             path.append(nxt)
    #                             visited.add((curr, nxt)); visited.add((nxt, curr))
    #                             prev, curr = curr, nxt
    #                         source_id = node_ids[path[0]]
    #                         target_id = node_ids[path[-1]]
    #                         edges.append((source_id, target_id, path))
    #                 for idx, (src, tgt, path) in enumerate(edges):
    #                     # Extract x/y coordinates
    #                     ys, xs = zip(*path)
    #                     ax.plot(xs, ys, color='green', linewidth=1)
    #                     # Label at midpoint
    #                     mid = len(path) // 2
    #                     y_mid, x_mid = path[mid]
    #                     ax.text(x_mid, y_mid, str(idx), color='blue', fontsize=5, ha='center', va='center')
    #                 for pix in junctions:
    #                     y, x = pix
    #                     ax.scatter(x, y, color='red', s=20)
    #                 plt.tight_layout()
    #                 plt.title('Edge-Node Skeleton Graph')
    #                 pdf.savefig()
    #                 plt.close()
    #             # Graph Density
    #             if val["id"] == 10 and val.get("value,0") ==1:
    #                 function_applied = True
    #                 img = io.imread(self.image_path)
    #                 gray = color.rgb2gray(img)
    #                 threshold = filters.threshold_otsu(gray)
    #                 binary = gray < threshold

    #                 skeleton = morphology.skeletonize(binary)
                    
    #                 h, w = skeleton.shape
    #                 adj_list = {}

    #                 graph = networkx.Graph()
    #                 graph.add_nodes_from(adj_list.keys())
                    
                                
    #                 def add_edge(g, a, b):
    #                     g.setdefault(a,[]).append(b)
    #                     g.setdefault(b,[]).append(a)
                        
    #                 for y in range(h):
    #                     for x in range(w):
    #                         if not skeleton[y, x]:
    #                             continue
    #                         for dy in [-1, 0, 1]:
    #                             for dx in [-1, 0, 1]:
    #                                 if dy == 0 and dx == 0:
    #                                     continue
    #                                 ny, nx = y + dy, x + dx
    #                                 if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
    #                                     add_edge(adj_list, (y, x), (ny, nx))
                                        
    #                 for u, neighbors in adj_list.items():
    #                     for v in neighbors:
    #                         if not graph.has_edge(u, v):
    #                             graph.add_edge(u, v)

    #                 graph_density = networkx.density(graph)
    #                 fig = plt.figure()
    #                 plt.title('Graph Density')
    #                 plt.axis('off')
    #                 plt.text(0.5, 0.5, str(graph_density), ha='center', va='center', size=24) 
    #                 pdf.savefig(fig)
    #                 plt.close() 
    #             # Average Degree
    #             if val["id"] == 11 and val.get("value,0") ==1:
    #                 function_applied = True
    #                 img = io.imread(self.image_path)
    #                 gray = color.rgb2gray(img)
    #                 threshold = filters.threshold_otsu(gray)
    #                 binary = gray < threshold

    #                 skeleton = morphology.skeletonize(binary)
                    
    #                 h, w = skeleton.shape
    #                 adj_list = {}

    #                 graph = networkx.Graph()
    #                 graph.add_nodes_from(adj_list.keys())
                    
                                
    #                 def add_edge(g, a, b):
    #                     g.setdefault(a,[]).append(b)
    #                     g.setdefault(b,[]).append(a)
                        
    #                 for y in range(h):
    #                     for x in range(w):
    #                         if not skeleton[y, x]:
    #                             continue
    #                         for dy in [-1, 0, 1]:
    #                             for dx in [-1, 0, 1]:
    #                                 if dy == 0 and dx == 0:
    #                                     continue
    #                                 ny, nx = y + dy, x + dx
    #                                 if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
    #                                     add_edge(adj_list, (y, x), (ny, nx))
                                        
    #                 for u, neighbors in adj_list.items():
    #                     for v in neighbors:
    #                         if not graph.has_edge(u, v):
    #                             graph.add_edge(u, v)
                                
    #                 deglist = [d for _, d in graph.degree()]
    #                 avdeg = sum(deglist)/len(deglist) if deglist else 0
    #                 fig = plt.figure()
    #                 plt.title('Average Degree')
    #                 plt.axis('off')
    #                 plt.text(0.5, 0.5, str(avdeg), ha='center', va='center', size=24) 
    #                 pdf.savefig(fig)
    #                 plt.close() 
    #             # Degree Heatmap
    #             if val["id"] == 12 and val.get("value,0") ==1:
    #                 function_applied = True
    #                 img = io.imread(self.image_path)
    #                 gray = color.rgb2gray(img)
    #                 threshold = filters.threshold_otsu(gray)
    #                 binary = gray < threshold
    #                 skeleton = morphology.skeletonize(binary)
                    
    #                 h, w = skeleton.shape
    #                 adj_list = {}

    #                 graph = networkx.Graph()
    #                 graph.add_nodes_from(adj_list.keys())
                    
                                
    #                 def add_edge(g, a, b):
    #                     g.setdefault(a,[]).append(b)
    #                     g.setdefault(b,[]).append(a)
                        
    #                 for y in range(h):
    #                     for x in range(w):
    #                         if not skeleton[y, x]:
    #                             continue
    #                         for dy in [-1, 0, 1]:
    #                             for dx in [-1, 0, 1]:
    #                                 if dy == 0 and dx == 0:
    #                                     continue
    #                                 ny, nx = y + dy, x + dx
    #                                 if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
    #                                     add_edge(adj_list, (y, x), (ny, nx))
                                        
    #                 for u, neighbors in adj_list.items():
    #                     for v in neighbors:
    #                         if not graph.has_edge(u, v):
    #                             graph.add_edge(u, v)
                                
    #                 degmap = np.zeros(skeleton.shape,dtype=float)
    #                 for (y,x), deg in graph.degree():
    #                     degmap[y,x] = deg
    #                 plt.figure(figsize=(5,5))
    #                 plt.imshow(gray,cmap='gray')
    #                 plt.imshow(degmap, cmap='magma', alpha=0.5)
    #                 plt.colorbar(label='Pixel Degree')
    #                 plt.title('Degree Heatmap')
    #                 plt.axis('off')
    #                 pdf.savefig()
    #                 plt.close()
    #             # Betweenness Centrality Heatmap
    #             if val["id"] == 13 and val.get("value,0") ==1:
    #                 function_applied = True
    #                 img = io.imread(self.image_path)
    #                 gray = color.rgb2gray(img)
    #                 threshold = filters.threshold_otsu(gray)
    #                 binary = gray < threshold
    #                 skeleton = morphology.skeletonize(binary)
                    
    #                 h, w = skeleton.shape
    #                 adj_list = {}
                        
    #                 def add_edge(g, a, b):
    #                     g.setdefault(a,[]).append(b)
    #                     g.setdefault(b,[]).append(a)
                        
    #                 for y in range(h):
    #                     for x in range(w):
    #                         if not skeleton[y, x]:
    #                             continue
    #                         for dy in [-1, 0, 1]:
    #                             for dx in [-1, 0, 1]:
    #                                 if dy == 0 and dx == 0:
    #                                     continue
    #                                 ny, nx = y + dy, x + dx
    #                                 if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
    #                                     add_edge(adj_list, (y, x), (ny, nx))
                                        
    #                 graph = networkx.Graph()
    #                 graph.add_nodes_from(adj_list.keys())
                    
    #                 for u, neighbors in adj_list.items():
    #                     for v in neighbors:
    #                         if not graph.has_edge(u, v):
    #                             graph.add_edge(u, v)
                                
    #                 bc_dict = networkx.betweenness_centrality(graph)
    #                 bc_map = np.zeros(skeleton.shape, dtype=float)
                    
    #                 for (y,x), score in bc_dict.items():
    #                     bc_map[y,x] = score
                        
    #                 plt.figure(figsize=(5,5))
    #                 plt.imshow(gray, cmap='gray')
    #                 plt.imshow(bc_map, cmap='magma', alpha=0.7)
    #                 plt.colorbar(label='Betweenness Centrality')
    #                 plt.title('Betweenness Centrality Heatmap')
    #                 plt.axis('off')
    #                 pdf.savefig()
    #                 plt.close()
    #             # # Edge Lengths and Widths CSV
    #             # if val["id"] == 14 and val.get("value,0") ==1:
    #             #     function_applied = True
    #             #     self.image_processor.edge_lenwid()
    #             # # Node Radii CSV
    #             # if val["id"] == 15 and val.get("value,0") ==1:
    #             #     function_applied = True
    #             #     self.image_processor.node_rad()

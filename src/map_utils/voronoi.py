'''
Based on the code from:
https://github.com/aislabunimi/ROSE2/blob/main/src/util/voronoi.py

'''


import math
from pathlib import Path
import sys
import matplotlib.colors
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from skan.csr import skeleton_to_csgraph
from skimage import img_as_bool
from skimage.morphology import skeletonize
from skimage.util import invert


def remove_2_neighbors(graph):
    old_edges = 0
    new_edges = len(graph.edges)
    precision = old_edges - new_edges
    while precision != 0:
        for node in graph.nodes:
            tmp = list(graph.neighbors(node))
            if len(tmp) == 2:
                graph.remove_edge(node, tmp[0])
                graph.remove_edge(node, tmp[1])
                graph.add_edge(tmp[0], tmp[1])
        old_edges = new_edges
        new_edges = len(graph.edges)
        precision = old_edges - new_edges
    graph.remove_nodes_from(list(nx.isolates(graph)))




def remove_1_neighbors(graph):
    old_edges = 0
    new_edges = len(graph.edges)
    precision = old_edges - new_edges
    while precision != 0:
        for node in graph.nodes:
            tmp = list(graph.neighbors(node))
            if len(tmp) == 1:
                graph.remove_edge(node, tmp[0])
        old_edges = new_edges
        new_edges = len(graph.edges)
        precision = old_edges - new_edges
    graph.remove_nodes_from(list(nx.isolates(graph)))


def remove_2_neighbors_one_cycle(graph):
    li = []
    for node in graph.nodes:
        tmp = list(graph.neighbors(node))
        if len(tmp) == 2:
            li.append(node)
    for n in li:
        tmp = list(graph.neighbors(n))
        if len(tmp) == 2:
            li.append(n)
            graph.remove_edge(n, tmp[0])
            graph.remove_edge(n, tmp[1])
            graph.add_edge(tmp[0], tmp[1])
    graph.remove_nodes_from(list(nx.isolates(graph)))


def remove_1_neighbors_one_cycle(graph):
    li = []
    for node in graph.nodes:
        tmp = list(graph.neighbors(node))
        if len(tmp) == 1:
            li.append(node)
    for el in li:
        n = list(graph.neighbors(el))
        graph.remove_edge(el, n[0])
    graph.remove_nodes_from(list(nx.isolates(graph)))


def remove_1_and_2_neighbors(graph):
    tb = 0
    tc = 0
    for node in graph.nodes:
        tmp = list(graph.neighbors(node))
        if len(tmp) == 2:
            tb += 1
            graph.remove_edge(node, tmp[0])
            graph.remove_edge(node, tmp[1])
            graph.add_edge(tmp[0], tmp[1])
        elif len(tmp) == 1:
            tc += 1
            graph.remove_edge(node, tmp[0])
    graph.remove_nodes_from(list(nx.isolates(graph)))

def setup_plot(size):
	plt.clf()
	plt.cla()
	plt.close('all')
	width = size[0]/100
	height = size[1]/100
	fig, ax = plt.subplots()
	fig.set_size_inches(width, height)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.axis('off')
	fig.add_axes(ax)
	return fig, ax

def plot_line(node_i, node_j, ax):
    x_coordinates = []
    y_coordinates = []
    x1 = int(node_i[1])
    x2 = int(node_j[1])
    y1 = int(node_i[0])
    y2 = int(node_j[0])
    x_coordinates.extend((x1, x2))
    y_coordinates.extend((y1, y2))
    ax.plot(x_coordinates, y_coordinates, color='k', linewidth=1)
    del x_coordinates[:]
    del y_coordinates[:]


def removed_isolated_cycles(graph: nx.Graph) -> nx.Graph:
    li = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest_cc = li[0]
    voronoi_graph = graph.subgraph(largest_cc).copy()
    voronoi_graph.remove_nodes_from(list(nx.isolates(voronoi_graph)))
    return voronoi_graph


def plot_voronoi(coordinates: list[tuple], voronoi_graph: nx.Graph, ax: plt.Axes, label, is_labelled, name, filepath:str):
    for edge in voronoi_graph.edges:
        plot_line(coordinates[edge[0]], coordinates[edge[1]], ax)
    for node in voronoi_graph.nodes:
        if is_labelled:
            col = (label[node][0] / 255, label[node][1] / 255, label[node][2] / 255)
            ax.scatter(coordinates[node][1], coordinates[node][0], c=matplotlib.colors.rgb2hex(col))
        else:
            ax.scatter(coordinates[node][1], coordinates[node][0])
    print('Plotting voronoi graph in: ', filepath + name + '.png')
    plt.savefig(filepath + name + '.png')


def evaluate_distance(node1, node2):
    distance = math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)
    return distance


def remove_close_nodes(graph: nx.Graph, coordinates, th):
    for node in graph.nodes:
        for n in graph.nodes:
            if n != node and evaluate_distance(coordinates[node], coordinates[n]) < th:
                graph = nx.contracted_nodes(graph, node, n, self_loops=False)
    return graph


def exist_path(color, paths, label):
    same_color = False
    for path in paths:
        for point in path:
            if label[point] == color:
                same_color = True
            else:
                same_color = False
                break
        if same_color is True:
            break
    return same_color


def compute_lines_direction(centers, v, u, n1, n2, directions):
    point1 = centers[v]
    point2 = centers[u]
    pt = [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]
    pt_n = [(n1[0] + n2[0]) / 2, (n1[1] + n2[1]) / 2]
    lines1 = []
    lines2 = []
    for direction in directions:
        m = math.tan(direction)
        c1 = point1[1] - m * point1[0]
        c2 = point2[1] - m * point2[0]
        lines1.append(c1)
        lines2.append(c2)
    max_distance = None
    direction = 0
    m_pt = None
    c_pt = None
    for i, line in enumerate(lines1):
        par_line = lines2[i]
        dist = abs(line - par_line) / math.sqrt(1 + math.tan(directions[i]) * math.tan(directions[i]))
        if max_distance is None or dist > max_distance:
            max_distance = dist
            direction = directions[i]
            m_pt = math.tan(directions[i])
            c_pt = pt_n[1] - m_pt * pt_n[0]
    return direction, m_pt, c_pt


def compute_intersections(x_min, x_max, y_min, y_max, m, c):
    points = []
    x1 = x_min
    y1 = m * x1 + c
    point1 = (int(x1), int(y1))
    # print('point1', point1)
    if y_min <= point1[1] <= y_max:
        points.append(point1)
    x2 = x_max
    y2 = m * x2 + c
    point2 = (int(x2), int(y2))
    # print('point2', point2)
    if y_min <= point2[1] <= y_max:
        points.append(point2)
    y3 = y_min
    x3 = (y3 - c) / m
    point3 = (int(x3), int(y3))
    # print('point3', point3)
    if x_min <= point3[0] <= x_max:
        points.append(point3)
    y4 = y_max
    x4 = (y4 - c) / m
    point4 = (int(x4), int(y4))
    # print('point4', point4)
    if x_min <= point4[0] <= x_max:
        points.append(point4)
    return points


def compute_most_distant(list_point, center, coordinates, pix_data, color):
    distance = 0
    point = None
    for p in list_point:
        pt = [int(coordinates[p][1]), int(coordinates[p][0])]
        dist = evaluate_distance(pt, center)
        if dist > distance and pix_data[pt[0], pt[1]] == color:
            point = (int(coordinates[p][1]), int(coordinates[p][0]))
    return point


def remove_close_nodes_1_neighbors(graph: nx.Graph, coordinates: list[tuple], thresh: float):
    old_edges = 0
    new_edges = len(graph.edges)
    precision = old_edges - new_edges
    while precision != 0:
        for node in graph.nodes:
            tmp = list(graph.neighbors(node))
            if len(tmp) == 1 and evaluate_distance(coordinates[node], coordinates[tmp[0]]) < thresh:
                graph.remove_edge(node, tmp[0])
        old_edges = new_edges
        new_edges = len(graph.edges)
        precision = old_edges - new_edges
    graph.remove_nodes_from(list(nx.isolates(graph)))


def remove_close_node_2_neighbors(graph: nx.Graph, coordinates: list[tuple], thresh: float):
    old_edges = 0
    new_edges = len(graph.edges)
    precision = old_edges - new_edges
    while precision != 0:
        for node in graph.nodes:
            tmp = list(graph.neighbors(node))
            if len(tmp) == 2 and evaluate_distance(coordinates[node], coordinates[tmp[0]]) < thresh and evaluate_distance(coordinates[node], coordinates[tmp[1]]) < thresh:
                graph.remove_edge(node, tmp[0])
                graph.remove_edge(node, tmp[1])
                graph.add_edge(tmp[0], tmp[1])
        old_edges = new_edges
        new_edges = len(graph.edges)
        precision = old_edges - new_edges
    graph.remove_nodes_from(list(nx.isolates(graph)))


def remove_close_nodes_1_neighbors_one_cycle(graph: nx.Graph, coordinates: list[tuple], thresh: float):
    li = []
    for node in graph.nodes:
        tmp = list(graph.neighbors(node))
        if len(tmp) == 1 and evaluate_distance(coordinates[node], coordinates[tmp[0]-1]) < thresh:
            li.append(node)
    for el in li:
        n = list(graph.neighbors(el))
        if len(n) == 1:
            graph.remove_edge(el, n[0])
    graph.remove_nodes_from(list(nx.isolates(graph)))


def reindex_nodes(graph: nx.Graph, coordinates: list[tuple]) -> tuple[nx.Graph, list[tuple]]:
    mapping = {}
    new_coordinates = []
    for new_idx, old_idx in enumerate(graph.nodes):
        mapping[old_idx] = new_idx
        new_coordinates.append(coordinates[old_idx])
    reindexed_graph = nx.relabel_nodes(graph, mapping)
    return reindexed_graph, new_coordinates


def compute_voronoi_graph(
        map_path: str | Path, 
        blur_radius: int, 
        min_node_distance: float, 
        plot_graph: bool = False, 
        name: str = '', 
        filepath: str =''
    ) -> tuple[nx.Graph, list[tuple]]:
    # --------------------------INITIALIZATION----------------------------------

    # load map
    copy = cv2.imread(str(map_path))
    im = copy.copy()


    im = cv2.blur(im, (blur_radius, blur_radius))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.threshold(im, 230, 255, cv2.THRESH_BINARY)[1]
    im = cv2.bitwise_not(im)

    initial_map = img_as_bool(im)

    # Invert image
    input_skeleton = invert(initial_map)

    # ------------------------------SKELETON-----------------------------------

    skeleton = skeletonize(input_skeleton)
    pixel_graph, coordinates = skeleton_to_csgraph(skeleton)
    x, y = coordinates
    coordinates = list(zip(x.tolist(), y.tolist()))
    # -------------------------------GRAPH-------------------------------------

    graph: nx.Graph = nx.from_scipy_sparse_array(pixel_graph)

    # ---------------------------PRUNING NODES----------------------------------

    # remove isolated part of graph
    voronoi_graph = removed_isolated_cycles(graph)


    remove_close_nodes_1_neighbors_one_cycle(voronoi_graph, coordinates, 40)
    remove_close_node_2_neighbors(voronoi_graph, coordinates, 60)
    remove_close_nodes_1_neighbors(voronoi_graph, coordinates, 30)

    if len(voronoi_graph.nodes) > 2:
        voronoi_graph = remove_close_nodes(voronoi_graph, coordinates, min_node_distance)

    graph.remove_nodes_from(list(nx.isolates(graph)))

    print('voronoi nodes:', len(voronoi_graph.nodes))

    # -------------------------------PLOT-------------------------------------

    graph, coordinates = reindex_nodes(voronoi_graph, coordinates)
    if plot_graph:
        _, ax = setup_plot([im.shape[1], im.shape[0]])
        ax.imshow(copy, cmap='gray')
        plot_voronoi(coordinates, graph, ax, 0, False, 'voronoi_graph_' + name, filepath=filepath)
    
    return graph, coordinates


if __name__ == '__main__':
    img_path = sys.argv[1]
    graph, coordinates = compute_voronoi_graph(img_path, 20, 20, plot_graph=True, name='test', filepath='')
    print(graph.nodes)
    print(list(list(graph.neighbors(i)) for i in graph.nodes))
    print(coordinates)

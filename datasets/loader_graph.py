import networkx as nx
from dp.graph_utils import generate_metagraph
import matplotlib.pyplot as plt
import numpy as np


class GraphLoader:
    def __init__(self, naming, dataset_name, num_classes):
        self.num_classes = num_classes - 1
        if naming == "EgoPER":
            if dataset_name == 'tea':
                self.graph_info = {
                    "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "edges": [(1, 2), (2, 4), (4, 5), (5, 6), (3, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
                }
            elif dataset_name == 'quesadilla':
                self.graph_info = {
                    "nodes": [1, 2, 3, 4, 5, 6, 7, 8],
                    "edges": [(1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6), (6, 7), (7, 8)]
                }
            elif dataset_name == 'oatmeal':
                self.graph_info = {
                    "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    "edges": [(1, 4), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 12), (12, 13), (13, 14), (14, 15), (15, 10), (10, 11), (15, 9)]
                }
            elif dataset_name == 'pinwheels':
                self.graph_info = {
                    "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                    "edges": [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 12), (12, 8), (8, 9), (9, 10), (10, 11), (11, 13)]
                }
            elif dataset_name == 'coffee':
                self.graph_info = {
                    "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    "edges": [(1, 2), (2, 13), (5, 13), (6, 7), (7, 8), (8, 12), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 3), (3, 4)]
                }
        elif naming == "CaptainCook4D":
            if dataset_name == "blenderbananapancakes":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                    "edges": [[14, 2],[7, 3],[13, 2],[2, 4],[11, 6],[5, 7],[12, 7],[1, 7],[9, 7],[6, 8],[8, 10],[3, 11],[10, 14]]
                }
            elif dataset_name == "breakfastburritos":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11],
                    "edges": [[6, 1], [1, 2], [2, 3], [8, 4], [11, 5], [9, 5], [3, 5], [4, 7], [10, 8], [5, 8], [2, 9], [2, 11]]
                }
            elif dataset_name == "broccolistirfry":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
                    "edges": [[11, 2], [9, 4], [16, 4], [13, 6], [23, 7], [17, 9], [21, 10], [19, 11], [18, 11], [5, 11], [20, 11], [6, 11], [3, 11], [1, 11], [4, 12], [8, 13], [22, 14], [7, 15], [17, 16], [14, 17], [24, 17], [10, 17], [2, 17], [25, 23], [12, 25]]
                }
            elif dataset_name == "capresebruschetta":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11],
                    "edges": [[11, 4], [7, 4], [2, 7], [5, 7], [6, 7], [10, 7], [1, 7], [3, 8], [9, 10], [8, 11]]
                }
            elif dataset_name == "cheesepimiento":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11],
                    "edges": [[4, 1], [4, 2], [10, 4], [5, 4], [9, 5], [7, 5], [4, 6], [8, 9], [3, 10], [6, 11], [1, 11], [2, 11]]
                }
            elif dataset_name == "cucumberraita":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11],
                    "edges": [[9, 2],[2, 5],[1, 6],[10, 6],[8, 6],[11, 6],[7, 6],[4, 6],[5, 10],[3, 10],[3, 1],[3, 4],[3, 7],[3, 8],[3, 11]]
                }
            elif dataset_name == "coffee":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                    "edges": [[2, 1], [16, 2], [3, 2], [4, 3], [12, 3], [5, 4], [7, 5], [14, 5], [1, 6], [11, 8], [15, 9], [6, 11], [10, 12], [13, 15], [9, 16]]
                }
            elif dataset_name == "dressedupmeatballs":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                    "edges": [[11, 2], [12, 2], [1, 3], [8, 5], [16, 6], [13, 7], [7, 8], [9, 12], [10, 12], [6, 12], [15, 12], [2, 13], [5, 14], [3, 15], [4, 16]]
                }
            elif dataset_name == "microwaveeggsandwich":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12],
                    "edges": [[3, 1], [10, 2], [9, 2], [1, 4], [6, 5], [8, 6], [2, 7], [11, 8], [12, 9], [5, 10], [4, 11]]
                }
            elif dataset_name == "microwavefrenchtoast":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11],
                    "edges": [[2, 1],[9, 2],[3, 2],[10, 3],[5, 3],[11, 5],[7, 6],[1, 7],[4, 8],[8, 11],[11, 10]]
                }
            elif dataset_name == "mugcake":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                    "edges": [[3, 1], [20, 1], [7, 1], [12, 2], [15, 3], [2, 5], [15, 7], [10, 8], [13, 9], [16, 12], [18, 13], [1, 14], [8, 14], [17, 15], [6, 15], [11, 15], [4, 15], [19, 16], [5, 18], [14, 19], [15, 20]]
                }
            elif dataset_name == "panfriedtofu":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
                    "edges": [[17, 2], [12, 2], [5, 2], [8, 3], [8, 4], [19, 5], [18, 6], [15, 7], [16, 8], [14, 9], [4, 10], [3, 10], [6, 11], [11, 13], [7, 14], [10, 15], [13, 16], [2, 18], [1, 19]]
                }
            elif dataset_name == "ramen":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                    "edges": [[9, 1], [7, 1], [12, 1], [1, 2], [8, 4], [6, 5], [4, 7], [15, 8], [4, 9], [14, 10], [3, 11], [13, 11], [4, 12], [10, 13], [5, 13], [11, 15]]
                }
            elif dataset_name == "spicedhotchocolate":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7],
                    "edges": [[3, 1],[7, 3],[4, 3],[2, 3],[6, 4],[6, 7],[6, 2],[5, 6]]
                }
            elif dataset_name == "tomatochutney":
                self.graph_info = {
                    # "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], # remove step 20
                    "edges": [[16, 1], [7, 2], [16, 3], [15, 5], [8, 6], [4, 6], [19, 8], [10, 9], [13, 10], [17, 12], [14, 13], [1, 14], [3, 14], [6, 15], [12, 16], [2, 16], [11, 17], [5, 18], [9, 19], [19, 4]]
                }
            elif dataset_name == "tomatomozzarellasalad":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9],
                    "edges": [[4, 1], [1, 2], [6, 3], [5, 3], [9, 3], [8, 3], [7, 4], [2, 5], [2, 6], [2, 8], [2, 9]]
                }
            elif dataset_name == "buttercorncup":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12],
                    "edges": [[4, 1], [2, 11], [11, 5], [5, 7], [7, 12], [7, 3], [12, 9], [3, 9], [9, 8], [8, 10], [8, 1], [10, 6], [1, 6]]
                }
            elif dataset_name == "herbomeletwithfriedtomatoes":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                    "edges": [[3, 8], [15, 14], [5, 14], [8, 14], [13, 12], [14, 1], [12, 1],[1, 6], [6, 10], [10, 2], [2, 7], [7, 4], [4, 11], [11, 9]],
                }
            elif dataset_name == "microwavemugpizza":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                    "edges": [[9, 8], [9, 6], [9, 1], [9, 13], [8, 11], [6, 11], [1, 11], [13, 11], [11, 14], [11, 7], [14, 10], [7, 10], [10, 3], [3, 4], [4, 12], [12, 2], [2, 5]]
                }
            elif dataset_name == "pinwheels":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
                    "edges": [[14, 12], [12, 11], [11, 17], [17, 10], [10, 6], [6, 3], [3, 15], [15, 9], [9, 13], [13, 5], [5, 16], [16, 4], [4, 7], [7, 1], [1, 8], [13, 2]]
                }
            elif dataset_name == "sautedmushrooms": # repetition for step 3, remove the first one, connect to 15 and 14
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                    "edges": [[13, 7], [2, 1], [1, 10], [10, 6], [7, 11], [8, 11], [6, 11], [11, 4], [4, 9], [9, 5], [5, 14], [5, 15], [14, 3], [15, 3], [3, 16], [3, 17], [16, 12], [17, 12]]
                }
            elif dataset_name == "scrambledeggs":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
                    "edges": [[19, 23], [20, 2], [7, 2], [10, 2], [4, 16], [12, 16], [23, 16], [1, 16], [6, 16], [2, 16], [16, 22], [16, 21], [22, 9], [21, 9], [9, 3], [9, 5], [3, 8], [5, 8], [8, 18], [8, 11], [18, 13], [11, 13], [13, 15], [15, 14], [14, 17]]
                }
            elif dataset_name == "spicytunaavocadowraps":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
                    "edges": [[16, 9], [9, 7], [3, 7], [15, 7], [12, 4], [4, 10], [10, 7], [7, 14], [7, 2], [11, 17], [5, 1], [17, 1], [14, 8], [2, 8], [1, 8], [8, 6], [6, 13]]
                }
            elif dataset_name == "zoodles":
                self.graph_info = {
                    "nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13],
                    "edges": [[12, 5], [9, 5], [5, 13],[13, 8], [8, 11], [11, 7], [11, 10], [11, 3], [11, 2], [7, 1], [10, 1], [3, 1], [2, 1], [1, 6], [6, 4]]
                }

        self.naming = naming
        self.dataset_name = dataset_name
        self.build_graphs()

    def insert_addition_nodes(self, metagraph):

        # pos = nx.planar_layout(metagraph)
        # nx.draw(metagraph, pos=pos, with_labels = True)
        # plt.savefig("metagraph.png")
        # plt.cla()

        edges = metagraph.edges()
        add_base = self.num_classes

        node_count = {}
        for i in range(1, self.num_classes+1):
            node_count[str(i)] = 0

        G = nx.DiGraph()
        for edge in edges:
            node1, node2 = edge

            node1_name = node1.split(',')[0]
            node2_name = node2.split(',')[0]
            
            if not G.has_node(node1):
                G.add_node(node1)
            
            add_node_name = str(add_base+int(node2_name))
            if not G.has_node(add_node_name):
                G.add_node(add_node_name)
            else:
                add_node_name = add_node_name+","+str(node_count[node2_name])
                G.add_node(add_node_name)
                node_count[node2_name] += 1
            if not G.has_node(node2):
                G.add_node(node2)

            G.add_edge(node1, add_node_name)
            G.add_edge(add_node_name, node2)

        add_nodes = []
        add_edges = []
        final_node_count = 1
        for node in G.nodes():
            node_name = node.split(',')[0]
            predecessors = [pred for pred in G.predecessors(node)]
            if len(predecessors) == 0:
                add_node_name = str(add_base+int(node_name))
                # there may be multiple starting points
                if G.has_node(add_node_name):
                    # add_node_name = add_node_name + ",-1"
                    add_node_name = add_node_name+","+str(node_count[node_name])
                    node_count[node_name] += 1
                add_nodes.append(add_node_name)
                add_edges.append((add_node_name, node))
            
            successors = [succ for succ in G.successors(node)]
            if len(successors) == 0:
                # assume there is only one end point
                add_node_name = "0"
                add_nodes.append(add_node_name)
                add_edges.append((node, add_node_name))
                final_node_count += 1

        for i in range(len(add_nodes)):
            G.add_node(add_nodes[i])
            G.add_edge(add_edges[i][0], add_edges[i][1])

        # pos = nx.planar_layout(G)
        # plt.figure(figsize=(12,12))
        # nx.draw(G, pos=pos, with_labels = True, node_size=60, font_size=8)
        # plt.savefig("gmetagraph.png")

        return G


    def build_graphs(self):
        node_info = self.graph_info["nodes"]
        edges_info = self.graph_info["edges"]
        
        G = nx.DiGraph()
        for i in node_info:
            G.add_node(i)
        for n1, n2 in edges_info:
            G.add_edge(n1, n2)

        self.graph_info["graph"] = G
        self.graph_info["metagraph"] = generate_metagraph(G)
        self.graph_info["gmetagraph"] = self.insert_addition_nodes(self.graph_info["metagraph"])

        # pos = nx.planar_layout(self.graph_info["graph"])
        # plt.figure(figsize=(12,12))
        # nx.draw(self.graph_info["graph"], pos=pos, with_labels = True, node_size=300, font_size=16)
        # plt.savefig("graph.png")
        # plt.cla()
        # pos = nx.planar_layout(self.graph_info["metagraph"])
        # plt.figure(figsize=(12,12))
        # nx.draw(self.graph_info["metagraph"], pos=pos, with_labels = True, node_size=300, font_size=16)
        # plt.savefig("metagraph.png")
        # plt.cla()
        # pos = nx.planar_layout(self.graph_info["gmetagraph"])
        # plt.figure(figsize=(16,16))
        # nx.draw(self.graph_info["gmetagraph"], pos=pos, with_labels = True, node_size=400, font_size=12)
        # plt.savefig("gmetagraph.png")

        # exit(0)

        node_list = self.graph_info["metagraph"].nodes()
        idx2node = {}
        idx = 0
        for node in node_list:
            idx2node[idx] = node
            idx += 1

        self.graph_info["idx2node"] = idx2node
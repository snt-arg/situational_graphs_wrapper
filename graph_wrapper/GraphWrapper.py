import numpy as np
import networkx as nx
import copy
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt
import itertools


SCORE_THR = 0.9


class GraphWrapper():
    def __init__(self, graph_def = None, graph_obj = None):
        if graph_def:
            graph = nx.Graph()
            graph.add_nodes_from(graph_def['nodes'])
            graph.add_edges_from(graph_def['edges'])
            self.name = graph_def['name']
            self.graph = graph

        elif graph_obj:
            self.name = "unknown"
            self.graph = graph_obj
        else:
            self.graph = nx.Graph()
            # print("GraphWrapper: definition of graph not provided. Empty graph created")


    def get_graph(self):
        return(self.graph)


    def categoricalMatch(self, G2, categorical_condition, draw = False):
        graph_matcher = isomorphism.GraphMatcher(self.graph, G2.get_graph(), node_match=categorical_condition, edge_match = lambda *_: True)
        matches = []
        if graph_matcher.subgraph_is_isomorphic():
            for subgraph in graph_matcher.subgraph_isomorphisms_iter():
                matches.append(subgraph)
                if draw:
                    plot_options = {
                        'node_size': 50,
                        'width': 2,
                        'with_labels' : True}
                    plot_options = self.define_draw_color_from_node_list(plot_options, subgraph, "blue", "orange")
                    self.draw("Graph {}".format(self.name), None, True)

        return matches ### TODO What should be returned?


    def matchByNodeType(self, G2, draw= False):
        categorical_condition = isomorphism.categorical_node_match(["type"], ["none"])
        matches = self.categoricalMatch(G2, categorical_condition, draw)
        matches_as_set_of_tuples = [set(zip(match.keys(), match.values())) for match in matches]
        # matches_as_list = [[[key, match[key]] for key in match.keys()] for match in matches]
        # print("GM: Found {} candidates after isomorphism and cathegorical in type matching".format(len(matches_as_set_of_tuples),))
        return matches_as_set_of_tuples


    # def matchIsomorphism(self, G1_name, G2_name):
    #     return isomorphism.GraphMatcher(self.graphs[G1_name], self.graphs[G2_name])


    def draw(self, fig_name = None, options = None, show = False):
        if not options:
            options = {'node_color': 'red', 'node_size': 50, 'width': 2, 'with_labels' : True}

        options = self.define_draw_position_option_by_attr(options)

        if fig_name:
            fig = plt.figure(fig_name, figsize=(200,200))
            ax = plt.gca()
            ax.clear()
        nx.draw(self.graph, **options)
        # plt.axis('on')
        # ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        if show:
            plt.show(block=False)
            plt.grid()
            plt.pause(0.2)

    
    def define_draw_color_from_node_list(self, options, node_list, unmatched_color = "red", matched_color = "blue"):
        if "node_color" not in options.keys():
            colors = dict(zip(self.graph.nodes(), [unmatched_color] * len(self.graph.nodes())))
        else:
            colors = dict(zip(self.graph.nodes(), options['node_color'])) 
            
        for origin_node in node_list:
            colors[str(origin_node)] = matched_color
        options['node_color'] = colors.values()
        return options


    def filter_graph_by_node_types(self, types):
        def filter_node_types_fn(node):
            return True if self.graph.nodes(data=True)[node]["type"] in types else False

        graph_filtered = GraphWrapper(graph_obj = nx.subgraph_view(self.graph, filter_node=filter_node_types_fn))
        return graph_filtered
    
    def filter_graph_by_edge_types(self, types):
        def filter_edge_types_fn(n1, n2):
            return True if self.graph[n1][n2]["type"] in types else False

        graph_filtered = GraphWrapper(graph_obj = nx.subgraph_view(self.graph, filter_edge=filter_edge_types_fn))
        return graph_filtered
    
    def filter_graph_by_node_attributes(self, attrs):
        def filter_node_attrs_fn(node):
            return attrs.items() <= self.graph.nodes(data=True)[node].items()

        graph_filtered = GraphWrapper(graph_obj = nx.subgraph_view(self.graph, filter_node=filter_node_attrs_fn))
        return graph_filtered
    
    def filter_graph_by_edge_attributes(self, attrs):
        def filter_edge_attrs_fn(n1, n2):
            return attrs.items() <= self.graph[n1][n2].items()

        graph_filtered = GraphWrapper(graph_obj = nx.subgraph_view(self.graph, filter_edge=filter_edge_attrs_fn))
        return graph_filtered

    def filter_graph_by_node_attributes_containted(self, attrs):
        def filter_node_attrs_fn(node):
            return all([True if attrs[key] in self.graph.nodes(data=True)[node][key] else False for key in attrs.keys()])

        graph_filtered = GraphWrapper(graph_obj = nx.subgraph_view(self.graph, filter_node=filter_node_attrs_fn))
        return graph_filtered
    

    def filter_graph_by_node_list(self, id_list):
        def filter_node_fn(node):
            return True if node in id_list else False 

        graph_filtered = GraphWrapper(graph_obj = nx.subgraph_view(self.graph, filter_node=filter_node_fn))
        return graph_filtered


    def stack_nodes_feature(self, node_list, feature):
        return np.array([self.get_attributes_of_node(key)[feature] for key in node_list]).astype(np.float64)


    def get_neighbourhood_graph(self, node_name):
        neighbours = self.graph.neighbors(node_name)
        filtered_neighbours_names = list([n for n in neighbours]) + [node_name]
        subgraph = GraphWrapper(graph_obj= self.graph.subgraph(filtered_neighbours_names))
        return(subgraph)


    def get_total_number_nodes(self):
        return(len(self.graph.nodes(data=True)))


    def add_subgraph(self, nodes_def, edges_def):
        self.add_nodes(nodes_def)
        self.add_edges(edges_def)

    def add_nodes(self, nodes_def):
        self.unfreeze()
        [self.graph.add_node(node_def[0], **node_def[1]) for node_def in nodes_def]

    def add_edges(self, edges_def):
        self.unfreeze()
        [self.graph.add_edge(edge_def[0], edge_def[1], **edge_def[2]) for edge_def in edges_def]

    def unfreeze(self):
        if nx.is_frozen(self.graph):
            self.graph = nx.Graph(self.graph)

    def define_draw_color_option_by_node_type(self, ):
        color_palette = {"floor" : "orange", "Infinite Room" : "cyan", "Finite Room" : "cyan", "Plane" : "orange", "origin" : "black"}
        color_palette.update({"room" : "cyan", "ws" : "orange"})
        type_list = [node[1]["type"] for node in self.graph.nodes(data=True)]
        colors = [color_palette[node_type] for node_type in type_list]

        return colors


    def define_draw_position_option_by_attr(self, options):
        if all(["draw_pos" in node[1].keys() for node in self.graph.nodes(data=True)]):
            pos = {}
            for node in self.graph.nodes(data=True):
                pos[node[0]] = node[1]["draw_pos"]
            options["pos"] = pos
        return options


    def define_node_size_option_by_combination_type_attr(self):
        if all(["combination_type" in node[1].keys() for node in self.graph.nodes(data=True)]):
            size = []
            for node in self.graph.nodes(data=True):
                if node[1]["combination_type"] == "group":
                    size.append(150)
                elif node[1]["combination_type"] == "pair":
                    size.append(50)
        else:
            size = None
        return size
    

    def define_node_linewidth_option_by_combination_type_attr(self):
        if all(["merge_lvl" in node[1].keys() for node in self.graph.nodes(data=True)]):
            linewidth = []
            for node in self.graph.nodes(data=True):
                if node[1]["merge_lvl"] == 1:
                    linewidth.append(1)
                else:
                    linewidth.append(0)
        else:
            linewidth = None
        return linewidth


    def filterout_unparented_nodes(self):
        new_graph = copy.deepcopy(self.graph)
        [new_graph.remove_node(node) for node in self.graph.nodes() if len(list([n for n in self.graph.neighbors(node)])) == 0]
        self.graph = new_graph

    def remove_nodes(self, node_IDs):
        self.graph.remove_nodes_from(node_IDs)

    def remove_edges(self, edge_IDs):
        self.graph.remove_edges_from(edge_IDs)
        
    def remove_all_edges(self):
        self.graph = nx.Graph(self.graph)
        edge_IDs = self.get_edges_ids()
        self.remove_edges(edge_IDs)

    def find_nodes_by_attrs(self, attrs):
        def test_fn(a,b):
            try:
                iter(b)
                return a in b
            except TypeError:
                return a == b
            else:
                return False
        nodes = [x for x,y in self.graph.nodes(data=True) if all(test_fn(attrs[attr], y[attr]) for attr in attrs.keys())]
        return nodes

    
    def set_node_attributes(self, attr_name, values_dict):
        nx.set_node_attributes(self.graph, values_dict, attr_name)

    def set_node_attributes_by_id_attr_dict(self, node_id, attr_dict):
        for key in attr_dict.keys():
            self.set_node_attributes(key, {node_id : attr_dict[key]})

    def get_attributes_of_node(self, node_id):
        return self.graph.nodes(data=True)[node_id]
    
    def get_attributes_of_all_nodes(self):
        return self.graph.nodes(data=True)
    
    def get_attributes_of_edge(self, edge_id): ### TODO Not working
        return self.graph.edges[edge_id[0],edge_id[1]]
    
    def get_attributes_of_all_edges(self):
        return self.graph.edges(data=True)

    def get_nodes_ids(self):
        return self.graph.nodes()
    
    def get_edges_ids(self):
        return self.graph.edges()
    
    def update_node_attrs(self, node_id, attrs):
        self.graph.nodes[node_id].update(attrs)

    def update_edge_attrs(self, edge_id, attrs):
        self.graph.edges[edge_id[0],edge_id[1]].update(attrs)

    def set_name(self, name):
        self.name = name

    def is_empty(self):
        return True if len(self.graph.nodes) == 0 else False
    
    def get_all_node_types(self):
        attributes_of_all_nodes = self.get_attributes_of_all_nodes()
        types = set()
        for node_attr in attributes_of_all_nodes:
            types.add(node_attr[1]["type"])

        return types
    
    def get_all_edge_types(self):
        attributes_of_all_edges = self.get_attributes_of_all_edges()
        types = set()
        for edge_attr in attributes_of_all_edges:
            types.add(edge_attr[2]["type"])

        return types
    
    def relabel_nodes(self, mapping = False, copy=False):
        if not mapping:
            mapping = dict(zip(self.get_nodes_ids(), range(len(self.get_nodes_ids()))))
        self.unfreeze()
        self.graph = nx.relabel_nodes(self.graph, mapping=mapping, copy=copy)

        return mapping
    
    def to_undirected(self, type = "smooth"):
        if type == "smooth":
            self.graph = self.graph.to_undirected()
        # elif type == "hard":
        #     self.graph = self.graph.to_undirected()
        #     current_edges = copy.deepcopy(list(self.get_edges_ids()))
        #     current_edges_sets = [set(edge) for edge in current_edges]
        #     to_remove_bool = [current_edges.count(edge) > 1 for edge in current_edges_sets]
        #     # for edge in current_edges:
        #     #     print(edge)
                

    def to_directed(self):
        self.graph = self.graph.to_directed()
        
    def stringify_node_ids(self):
        raw_node_ids = list(self.get_nodes_ids())
        str_node_ids = map(str, raw_node_ids)
        mapping = dict(zip(raw_node_ids, str_node_ids))
        self.relabel_nodes(mapping)

    def find_recursive_simple_cycles(self):
        graph = copy.deepcopy(self.graph)
        graph = graph.to_directed()
        simple_cycles = nx.recursive_simple_cycles(graph)
        return simple_cycles
    
    def translate_attr_all_nodes(self, attr_name, addition):
        values_dict = {}
        for node_id in self.get_nodes_ids():
            attr_value = self.get_attributes_of_node(node_id)[attr_name]
            values_dict[node_id] = attr_value + addition
        self.set_node_attributes(attr_name, values_dict)

    def merge_graph(self, new_graph):
        new_nx = nx.compose(self.graph,new_graph.graph)
        return GraphWrapper(graph_obj = new_nx)
    
    def is_directed(self):
        return self.graph.is_directed()
    
    def clone(self):
        return copy.deepcopy(self)
    
    def edges_of_node(self, node_id):
        return self.graph.edges(node_id)
    
    def to_file(self, path):
        self.graph.save(path)
    
    # def make_fully_connected(self):
    #     nodes_IDs = list(self.get_nodes_ids())
    #     current_edges = set(self.get_edges_ids())
    #     new_edges = set(itertools.combinations(nodes_IDs,2))
    #     new_edges = list(np.setdiff1d(new_edges, current_edges, assume_unique=False))
    #     self.graph.add_edges_from(new_edges)
    #     SDFasdf



    # ## Geometry functions

    # def planeIntersection(self, plane_1, plane_2, plane_3):
    #     normals = np.array([plane_1[:3],plane_2[:3],plane_3[:3]])
    #     # normals = np.array([[1,0,0],[0,1,0],[0,0,1]])
    #     distances = -np.array([plane_1[3],plane_2[3],plane_3[3]])
    #     # distances = -np.array([5,7,9])
    #     return(np.linalg.inv(normals).dot(distances.transpose()))


    # def computePlanesSimilarity(self, walls_1_translated, walls_2_translated, thresholds = [0.001,0.001,0.001,0.001]):
    #     differences = walls_1_translated - walls_2_translated
    #     conditions = differences < thresholds
    #     asdf

    
    # def computePoseSimilarity(self, rooms_1_translated, rooms_2_translated, thresholds = [0.001,0.001,0.001,0.001,0.001,0.001]):
    #     differences = rooms_1_translated - rooms_2_translated
    #     conditions = differences < thresholds
    #     asdf


    # def computeWallsConsistencyMatrix(self, ):
    #     pass


    # def changeWallsOrigin(self, original, main1_wall_i, main2_wall_i):
    #     # start_time = time.time()
    #     original[:,2] = np.array(np.zeros(original.shape[0]))   ### 2D simplification
    #     normalized = original / np.sqrt(np.power(original[:,:-1],2).sum(axis=1))[:, np.newaxis]
    #     intersection_point = self.planeIntersection(normalized[main1_wall_i,:], normalized[main2_wall_i,:], np.array([0,0,1,0]))

    #     #### Compute rotation for new origin
    #     z_normal = np.array([0,0,1]) ### 2D simplification
    #     x_axis_new_origin = np.cross(normalized[main1_wall_i,:3], z_normal)
    #     rotation = np.array((x_axis_new_origin,normalized[main1_wall_i,:3], z_normal))

    #     #### Build transform matrix
    #     rotation_0 = np.concatenate((rotation, np.expand_dims(np.zeros(3), axis=1)), axis=1)
    #     translation_1 = np.array([np.concatenate((intersection_point, np.array([1.0])), axis=0)])
    #     full_transformation_matrix = np.concatenate((rotation_0, -translation_1), axis=0)

    #     #### Matrix multiplication
    #     transformed = np.transpose(np.matmul(full_transformation_matrix,np.matrix(np.transpose(original))))
    #     transformed_normalized = transformed / np.sqrt(np.power(transformed[:,:-1],2).sum(axis=1))
    #     # print("Elapsed time in geometry computes: {}".format(time.time() - start_time))
    #     return transformed_normalized


    # def checkWallsGeometry(self, graph_1, graph_2, match):
    #     start_time = time.time()
    #     match_keys = list(match.keys())
    #     room_1 = np.array([graph_1.nodes(data=True)[key]["pos"] for key in match_keys])
    #     room_2 = np.array([graph_2.nodes(data=True)[match[key]]["pos"] for key in match_keys])
    #     room_1_translated = self.changeWallsOrigin(room_1,0,1)
    #     room_2_translated = self.changeWallsOrigin(room_2,0,1)
    #     scores = self.computePlanesSimilarity(room_1_translated,room_2_translated)
    #     return False

    def nx_to_hetero(self):
        import torch
        from torch_geometric.data import HeteroData

        graph = copy.deepcopy(self.graph)        
        hetero_data = HeteroData()

        # Identify node types and initialize storage
        node_types = set([node[1]['type'] for node in graph.nodes(data=True)])
        for node_type in node_types:
            hetero_data[node_type].x = []

        # Identify edge types and initialize storage
        edge_types = set((data['type'] for _, _, data in graph.edges(data=True)))
        print(f"dbg edge_types {edge_types}")
        for edge_type in edge_types:
            example_src_id, example_dst_id, _ = list(self.filter_graph_by_edge_types(edge_type).get_attributes_of_all_edges())[0]
            src_type = self.get_attributes_of_node(example_src_id)["type"]
            dst_type = self.get_attributes_of_node(example_dst_id)["type"]
            hetero_data[(src_type, edge_type, dst_type)].edge_index = []
            hetero_data[(src_type, edge_type, dst_type)].edge_attr = []

        # Add nodes to hetero_data
        for node, data in graph.nodes(data=True):
            node_type = data['type']
            if 'x' in data:
                hetero_data[node_type].x.append(data['x'])
            else:
                hetero_data[node_type].x.append([0.0])  # Default feature if not provided

        # Convert lists to tensors
        for node_type in node_types:
            hetero_data[node_type].x = torch.tensor(hetero_data[node_type].x, dtype=torch.float)

        # Add edges to hetero_data
        for src, dst, data in graph.edges(data=True):
            edge_type = data['type']
            src_type, rel_type, dst_type = edge_type.split('_') ### TODO Fix edge types
            print(f"dbg hetero_data {hetero_data.edge_types}")
            hetero_data[(src_type, rel_type, dst_type)].edge_index.append([src, dst])
            if 'edge_attr' in data:
                hetero_data[(src_type, rel_type, dst_type)].edge_attr.append(data['edge_attr'])
            else:
                hetero_data[(src_type, rel_type, dst_type)].edge_attr.append([0.0])  # Default edge attribute if not provided

        # Convert lists to tensors
        for edge_type in edge_types:
            src_type, rel_type, dst_type = edge_type.split('_')
            edge_index = torch.tensor(hetero_data[(src_type, rel_type, dst_type)].edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(hetero_data[(src_type, rel_type, dst_type)].edge_attr, dtype=torch.float)
            hetero_data[(src_type, rel_type, dst_type)].edge_index = edge_index
            hetero_data[(src_type, rel_type, dst_type)].edge_attr = edge_attr

        return hetero_data
    
    def nx_to_homo(self):
        import torch
        from torch_geometric.data import Data

        graph = copy.deepcopy(self.graph)    
        all_x = []
        all_edge_index = []
        all_edge_attrs = []
        node_types = []
        edge_types = []

        # Create mappings for node and edge types
        node_type_mapping = {type: i for i, type in enumerate(set(nx.get_node_attributes(graph, 'type').values()))}
        edge_type_mapping = {type: i for i, type in enumerate(set(data['type'] for _, _, data in graph.edges(data=True)))}

        # Add nodes to all_x and node_types
        for node, data in graph.nodes(data=True):
            if 'x' in data:
                all_x.append(data['x'])
            else:
                all_x.append([0.0])  # Default feature if not provided
            node_types.append(node_type_mapping[data['type']])

        # Add edges to all_edge_index, edge_attrs, and edge_types
        for src, dst, data in graph.edges(data=True):
            all_edge_index.append([src, dst])
            if 'edge_attr' in data:
                all_edge_attrs.append(data['edge_attr'])
            else:
                all_edge_attrs.append([0.0])  # Default edge attribute if not provided
            edge_types.append(edge_type_mapping[data['type']])

        # Convert lists to tensors
        x = torch.tensor(all_x, dtype=torch.float)
        edge_index = torch.tensor(all_edge_index, dtype=torch.long).t().contiguous()
        edge_attrs = torch.tensor(all_edge_attrs, dtype=torch.float)
        node_type_indices = torch.tensor(node_types, dtype=torch.long)
        edge_type_indices = torch.tensor(edge_types, dtype=torch.long)

        # Create a new Data object with the concatenated features and additional attributes
        homo_data = Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attrs, 
            node_type=node_type_indices, 
            edge_type=edge_type_indices
        )

        return homo_data

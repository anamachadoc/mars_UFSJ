from pyvis.network import Network
import networkx as nx
from collections import Counter
import os
import plotly.graph_objects as go
import igraph as ig
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Graph():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.graph = self.__init_graph()
        os.makedirs(self.savepath, exist_ok=True)
        
    def __init_graph(self):
        G = nx.DiGraph() if self.directed else nx.Graph()
        users = {}
        for filename in os.listdir(self.datapath):
            with open(os.path.join(self.datapath, filename), 'r') as arquivo:  
                followers = [linha.strip() for linha in arquivo.readlines()]
            users[filename[:-4]] = followers
        edges = [(follower, user) for user, followers in users.items() for follower in followers]
        G.add_edges_from(edges)
        return G
    
    def get_nodes(self):
        return list(self.graph.nodes())
    
    def get_egdes(self):
        return list(self.graph.nodes())
    
    def get_num_nodes(self):
        return self.graph.number_of_nodes()
    
    def get_num_edges(self):
        return self.graph.number_of_edges()
    
    def get_nodes_degree(self):
        if self.directed:
            nodes_degree = {'in': dict(self.graph.in_degree()), 'out': dict(self.graph.out_degree())}
        else:
            nodes_degree = dict(self.graph.degree())
        return nodes_degree
    
    def get_clustering_coefficient(self):
        return [nx.clustering(self.graph, node, weight=None) for node in self.get_nodes()]
        
    def get_mean_clustering(self):
        return round(np.mean(self.get_clustering_coefficient()), 4)
    
    def save_clustering_coefficient(self):
        clustering_coefficient = self.get_clustering_coefficient()
        df = pd.DataFrame({'User': self.get_nodes(), 'Clustering nodes': clustering_coefficient})
        df = df.sort_values(by='Clustering nodes', ascending=False)
        df.to_csv(f'{self.savepath}/clustering_coefficient.csv', index=False)
        
    def save_nodes_degree(self):
        nodes_degree = self.get_nodes_degree()
        if self.directed:
            dict = {'User': [], 'Degree In': [], 'Degree Out': []}
            for user in self.get_nodes():
                dict['User'].append(user)
                dict['Degree In'].append(nodes_degree['in'][user])
                dict['Degree Out'].append(nodes_degree['out'][user])
            df = pd.DataFrame(dict)
            df = df.sort_values(by='Degree In', ascending=False)
            df.to_csv(f'{self.savepath}/nodes_degree_sorted_by_degree_in.csv', index=False)
            df = df.sort_values(by='Degree Out', ascending=False)
            df.to_csv(f'{self.savepath}/nodes_degree_sorted_by_degree_out.csv', index=False)
        else:
            df = pd.DataFrame(list(nodes_degree.items()), columns=['User', 'Degree'])
            df = df.sort_values(by='Degree', ascending=False)
            df.to_csv(f'{self.savepath}/nodes_degree.csv', index=False)
           
    def is_directed(self):
        return self.graph.is_directed()

    def plot_degree_distribution(self, scale):
        if self.is_directed():
            function = self.graph.in_degree()
            self.__degree_distribution(scale, function, 'Degree in distribution', '#4169E1')
            function = self.graph.out_degree()
            self.__degree_distribution(scale, function, 'Degree out distribution', '#FF6347')
        else:
            function = self.graph.degree()
            self.__degree_distribution(scale, function, 'Degree distribution', '#4169E1')
            
    def __degree_distribution(self, scale, function, title, color):
        degrees = dict(function)
        degree_counts = Counter(degrees.values())
        
        d_raw = list(degree_counts.keys())
        pk_raw = [count / self.get_num_nodes() for count in degree_counts.values()]
        d, pk = zip(*[(k, p) for k, p in zip(d_raw, pk_raw) if k > 0 and p > 0])
    
        fig = go.Figure(data=go.Scatter(x=d, y=pk, mode='markers', marker=dict(color=color, size=7)))
        fig.update_layout(
            template='simple_white',
            title=f'{title} - {scale}',
            xaxis_title='k',
            yaxis_title='P(k)',
            showlegend=False
        )
        if scale == 'log':
            fig.update_layout(
                xaxis_type="log",  
                yaxis_type="log",  
                xaxis=dict(
                    exponentformat="E",  
                    tickvals=[10**i for i in range(int(np.log10(min(d))), int(np.log10(max(d)))+1)],
                    ticktext=[self.__scientific_format(i) for i in [10**i for i in range(int(np.log10(min(d))), int(np.log10(max(d)))+1)]]),
                yaxis=dict(
                    exponentformat="E", 
                    tickvals=[10**i for i in range(int(np.log10(min(pk))), int(np.log10(max(pk)))+1)],
                    ticktext=[self.__scientific_format(i) for i in [10**i for i in range(int(np.log10(min(pk))), int(np.log10(max(pk)))+1)]]))
        fig.write_image(f'{self.savepath}/{title}_{scale}.png')
        
    def __scientific_format(self, value):
        exponent = np.log10(value)  
        exponent_rounded = round(exponent)
        return f"10<sup>{exponent_rounded}</sup>"

    def get_degree_centrality(self, centrality):
        return [(node, round(value, 4)) for node, value in centrality.items()]
       
    def centrality(self, type):
        if type == 'degree centrality':
            if self.is_directed():
                in_centrality = nx.in_degree_centrality(self.graph)
                self.save_centrality(self.get_degree_centrality(in_centrality), f'in {type}')
                out_centrality = nx.out_degree_centrality(self.graph)
                self.save_centrality(self.get_degree_centrality(out_centrality), f'out {type}')
            else:
                centrality = nx.degree_centrality(self.graph)
                self.save_centrality(self.get_degree_centrality(centrality), type)
        elif type == 'eigenvector':
            cetrality = nx.eigenvector_centrality(self.graph)
            self.save_centrality(self.get_degree_centrality(cetrality), type)
        else:
            print(f"{type} not implemented. Choose between 'degree centrality' and 'eigenvector")
                
    def save_centrality(self, centrality, type):
        centrality_df = pd.DataFrame(centrality, columns=['vertice', type])
        centrality_df = centrality_df.sort_values(by=type, ascending=False)
        centrality_df.to_csv(f'{self.savepath}/{type}.csv', index=False)
        
    def plot_graph_html(self):
            nt = Network(
            notebook=True,
            cdn_resources='remote',
            height="750px",
            width="100%",
            bgcolor="white",
            neighborhood_highlight=True,
            select_menu=True,
            font_color='black')
            nt.from_nx(self.graph)
            for node in nt.nodes:
                node['size'] = 10 
            for edge in nt.edges:
                edge['color'] = 'rgba(200, 200, 200, 0.3)' 
            nt.barnes_hut(
                gravity=-1500,
                central_gravity=0.3,
                spring_length=250,
                spring_strength=0.001,
                damping=0.09,
                overlap=0)
            nt.show(f'{self.savepath}/graph.html')

    def plot_graph_default(self):
        G_ig = ig.Graph(directed=self.is_directed())
        G_ig.add_vertices(len(self.graph.nodes))
        nx_nodes = list(self.graph.nodes)
        nx_id_to_ig_id = {node: i for i, node in enumerate(nx_nodes)}
        edges = [(nx_id_to_ig_id[u], nx_id_to_ig_id[v]) for u, v in self.graph.edges()]
        G_ig.add_edges(edges)
        layout = G_ig.layout("fr") 
        visual_style = {
        "layout": layout,
        "vertex_size": 10,  
        "vertex_color": "skyblue",  
        "vertex_label": None,  
        "edge_color": "gray",  
        "edge_width": 1,  
        "edge_opacity": 0.2, 
        "bbox": (800, 800),  
        "margin": 20  
        }
        ig.plot(G_ig, f"{self.savepath}/graph.png", **visual_style)
        
    def __plot_by_centrality(self, centrality, title, mult=5):
        G_ig = ig.Graph(directed=self.is_directed())
        G_ig.add_vertices(len(self.graph.nodes))
        nx_nodes = list(self.graph.nodes)
        nx_id_to_ig_id = {node: i for i, node in enumerate(nx_nodes)}
        edges = [(nx_id_to_ig_id[u], nx_id_to_ig_id[v]) for u, v in self.graph.edges()]
        G_ig.add_edges(edges)
        centrality_values = [centrality[node] for node in nx_nodes]
        scaler = MinMaxScaler(feature_range=(1, 10))
        normalized_centrality = scaler.fit_transform(np.array(centrality_values).reshape(-1, 1)).flatten()
        normalized_centrality = [round(x, 1) for x in normalized_centrality]
        color_map = {
            1: '#87CEEB',
            2: '#00BFFF',
            3: '#1E90FF',
            4: '#6495ED',
            5: '#4169E1',
            6: '#0000FF',
            7: '#0000CD',
            8: '#00008B',
            9: '#000080',
            10: '#191970'}
        node_sizes = {}
        node_colors = {}
        for i, node in enumerate(nx_nodes):
            size = normalized_centrality[i] * mult if normalized_centrality[i] * mult > 1 else 1
            node_sizes[node] = size
            node_colors[node] = color_map[int(normalized_centrality[i])] 
        G_ig.vs["size"] = [node_sizes[node] for node in nx_nodes]
        G_ig.vs["color"] = [node_colors[node] for node in nx_nodes]
        G_ig.vs["label"] = [str(node) for node in nx_nodes]  
        layout = G_ig.layout("fr") 
        visual_style = {
            "layout": layout,
            "vertex_size": G_ig.vs["size"], 
            "vertex_color": G_ig.vs["color"],  
            "vertex_label": None, 
            "edge_color": "gray",  
            "edge_width": 1,  
            "edge_opacity": 0.2, 
            "bbox": (800, 800),  
            "margin": 20  }
        ig.plot(G_ig, f"{self.savepath}/{title}.png", **visual_style)

    def plot_graph_by_centrality(self, type):
        if type == 'degree centrality':
            if self.is_directed():
                in_centrality = nx.in_degree_centrality(self.graph)
                self.__plot_by_centrality(in_centrality, 'Degree in centrality')
                out_centrality = nx.out_degree_centrality(self.graph)
                self.__plot_by_centrality(out_centrality, 'Degree out centrality')
            else:
                centrality = nx.degree_centrality(self.graph)
                self.__plot_by_centrality(centrality, 'Degree centrality')
        elif type == 'eigenvector':
            centrality = nx.eigenvector_centrality(self.graph)
            self.__plot_by_centrality(centrality, 'Eigenvector centrality')
           
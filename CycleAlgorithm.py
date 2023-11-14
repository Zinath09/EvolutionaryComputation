import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
from utils import *
from Algorithm import Algorithm


class CycleAlgorithm(Algorithm):
    def __init__(self, data) -> None:
        super().__init__(data)

        self.total_cost = 0
        self.cur_tour = []
        self.edge_distances = []

    
    def starting_solution(self, starting_node=0, second_node=1):
        self.cur_tour = [[starting_node,second_node], [second_node,starting_node]]
        self.unvisited.remove(starting_node)
        self.unvisited.remove(second_node)
        self.total_cost = 2 * self.node_distances[starting_node][second_node]
        if self.is_cost:
            self.total_cost += \
                + self.cost_list[starting_node] \
                + self.cost_list[second_node] 
        self.edge_distances = self.count_edge_distances_to_unvisited_nodes(self.cur_tour)

    def add_to_tour(self, edge, new_n):        
        self.cur_tour.remove(edge)
        self.cur_tour.append([edge[0], new_n])
        self.cur_tour.append([new_n, edge[1]])
        self.unvisited.remove(new_n)
        self.total_cost += self.count_diff(edge, new_n)
        return [edge[0], new_n], [new_n, edge[1]]
    
    def count_diff(self,edge, new_n,):
        start_n = edge[0]
        end_n = edge[1]
        cost_diff = -self.node_distances[start_n][ end_n] \
            + self.node_distances[start_n][ new_n] \
            + self.node_distances[new_n][end_n] 
        if self.is_cost:
            cost_diff+= self.cost_list[new_n]
        return cost_diff
    
    def choose_removed_edge_index(self, new_node):
        edge_index = np.argmin(self.edge_distances[new_node], axis=0)
        return edge_index
    

    def count_edge_distances_to_unvisited_nodes(self,edges):
        matrix= np.zeros((self.NR_NODES,len(edges)))
        for new_node in self.unvisited:
            for i,edge in enumerate(edges):
                matrix[new_node][i]=self.count_diff(edge, new_node)
        return matrix
    

    def update_edge_distances(self, added_edges, removed_edge_index = False):
        added_edges_matrix = self.count_edge_distances_to_unvisited_nodes(added_edges)

        # assert(len(self.cur_tour)==2, 'This function comes after updating cur_nodes')
        if len(self.edge_distances) == 0:
            final_edge_distances = added_edges
        elif removed_edge_index == False:
            final_edge_distances = np.concatenate([self.edge_distances, added_edges_matrix], axis = 1)
 
        elif removed_edge_index == 0: #first edge
            reused_dist_2 = self.edge_distances[:,removed_edge_index+1:]
            final_edge_distances = np.concatenate([reused_dist_2, added_edges_matrix], axis = 1)

        elif removed_edge_index == len(self.cur_tour)-1: #last edge
            reused_dist_1 = self.edge_distances[:,: removed_edge_index]
            final_edge_distances = np.concatenate([reused_dist_1, added_edges_matrix], axis = 1)

        else:
            reused_dist_1 = self.edge_distances[:,: removed_edge_index]
            reused_dist_2 = self.edge_distances[:,removed_edge_index+1:]
            final_edge_distances = np.concatenate([reused_dist_1,reused_dist_2, added_edges_matrix], axis = 1)
        
        self.edge_distances = final_edge_distances

    def add_and_update_node(self, new_node):
        edge_index = self.choose_removed_edge_index(new_node)
        edge = self.cur_tour[edge_index]
        new_edges = self.add_to_tour(edge, new_node)
        self.update_edge_distances(new_edges, edge_index)

    def next_node(self):
        a = np.min(self.edge_distances, axis = 1)
        mask_array = np.zeros(self.NR_NODES, dtype=bool)
        mask_array[np.array(self.unvisited)] = True
        a[~mask_array] = np.inf
        return np.argmin(a)
    
    def create_cur_tour_and_update(self, lista):
        self.cur_tour = self.create_cur_tour_from_list(lista)
        
        for visited in np.unique(self.cur_tour):
            self.unvisited.remove(visited)
        self.update_edge_distances(self.cur_tour)

    def create_cur_tour_from_list(self, lista):
        lenght = len(lista)
        total_cost = 0
        edges = []
        for i in range(lenght):
            a = lista[i]
            b = lista[(i+1)%lenght]
            dist = self.node_distances[a][b]
            cost = self.cost_list[i]
            total_cost += dist + cost 
            edges.append([a,b])
            
        return edges
    

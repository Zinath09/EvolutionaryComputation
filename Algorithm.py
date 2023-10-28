import numpy as np
from utils import *

class Algorithm():
    
    def __init__(self, data) -> None:
        self.NR_NODES = len(data)

        self.unvisited = list(range(self.NR_NODES))

        self.is_cost = len(data[0]) == 3

        self.create_dist_matrix_and_cost(data)
        
    
    def starting_solution(self, starting_node = 0, second_node = 1):
        pass

    def add_to_tour(self):
        pass

    def count_diff(self):
        pass
    
    def create_dist_matrix_and_cost(self,data):
        node_distances = np.zeros((self.NR_NODES,self.NR_NODES))
        cost_list = np.zeros(self.NR_NODES)
        for i in range(self.NR_NODES):
            for j in range(i,self.NR_NODES):
                dist = Euclidian_distance(data[i][:2],data[j][:2])
                if self.is_cost:
                    cost_list[i] = data[i][2]
                node_distances[i][j] = dist
                node_distances[j][i] = dist
                # node_distances[i][i] = np.inf
        self.node_distances = node_distances
        if self.is_cost:
            self.cost_list = cost_list
            

    



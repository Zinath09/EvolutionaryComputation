import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
from utils import *
from Algorithm import Algorithm


class GreedyAlgorithm(Algorithm):
    def __init__(self, data) -> None:
        super().__init__(data)

        self.total_cost = 0
        self.cur_tour = []
        self.endings
        self.cost_matrix
        
    
    def starting_solution(self, starting_node=None):
        self.endings= [starting_node]

        if self.is_cost:
            self.cost_matrix = self.node_distances + self.cost_list.T
        else:
            self.cost_matrix = self.node_distances

        min, ind = get_min_index(self.cost_matrix[starting_node])

        self.endings.append(ind)

        self.total_cost = self.node_distances[starting_node, ind]

        if self.is_cost:
            self.total_cost =+ \
                self.cost_list[starting_node]+self.cost_list[ind]

        self.cost_matrix[:, self.endings] =  np.inf
        self.cur_tour = [[starting_node, ind]]

    def add_to_tour(self, previous_node, new_node):        
        self.endings.remove(previous_node)
        self.endings.append(new_node)

        self.cur_tour.append([previous_node, new_node])
        self.cost_matrix[:, new_node] =  np.inf
        return [previous_node, new_node]
    
    def count_diff(self, previous_node, new_node):
        cost_diff = -self.node_distances[previous_node][new_node]
        if self.is_cost:
            cost_diff+= self.cost_list[new_node]
        return cost_diff
    
    def next_node(self):
        
        a = np.min(self.cost_matrix[self.endings[0]], axis = 0)
        b = np.min(self.cost_matrix[self.endings[1]], axis = 0)

        if a<=b:
            return 0, np.argmin(self.cost_matrix[self.endings[0]])
        else:
            return 1, np.argmin(self.cost_matrix[self.endings[1]])    
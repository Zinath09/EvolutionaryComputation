import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import random
from utils import *
from algorithms import *
from Algorithm import Algorithm
from CycleAlgorithm import CycleAlgorithm




data = get_data('TSPD.csv')
data = np.array([(0, 0), (5, 3), (1, 4), (3, 1), (7, 3), (2,5), (4,4)]) * 100
alg = CycleAlgorithm(data)
alg.starting_solution(0,1)
alg.add_and_update_node(4)
while len(alg.unvisited) != 0:
    new_node = alg.next_node()
    edge_index = alg.choose_removed_edge_index(new_node)
    edge = alg.cur_tour[edge_index]
    print(new_node, edge)

    new_edges = alg.add_to_tour(edge, new_node)
    alg.update_edge_distances(new_edges, edge_index)
    # plotMap(data, alg.cur_tour, cost = False)



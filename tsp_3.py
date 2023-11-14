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
data = np.array([(0, 0, 0), (5, 3,0 ), (1, 4,0), (3, 1,0), (7, 3,0), (2,5,0), (4,4, 0)]) * 100

alg = CycleAlgorithm(data)


distance_matrix = alg.node_distances



lista = [0,4,2]
alg.create_cur_tour_and_update(lista)
# plotMap(data, alg.cur_tour)
unvisited = [x for x in range(len(data))]
for i in lista:
    unvisited.remove(i)

for i in range(30):
    # lista, unvisited = swap_unvisited_and_visited(lista, unvisited)
    find_first_better(lista, 2, unvisited)
    

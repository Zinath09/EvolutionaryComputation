import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
import csv 

def plotMap(nodes, edges=[], colors = False, cost = True):
    fig=plt.figure(figsize=(6,6), dpi= 100, facecolor='w', edgecolor='k')
    
    for e in (edges):
        start, end = e
        X = [nodes[start][0], nodes[end][0]]
        
        Y = [nodes[start][1], nodes[end][1]]
        plt.plot(X, Y, lw=1, ls="-", marker="", color = 'gray' )
        
    X = [c[0] for c in nodes]
    Y = [c[1] for c in nodes]
    if cost:
        S = [c[2] for c in nodes]
        plt.scatter(X, Y, c = S, cmap="grey")
    else:
        plt.scatter(X, Y, cmap="grey") 
    for i in range(len(nodes)):
        plt.annotate(i, (X[i], Y[i]+0.2))
    plt.show()


def get_min_value(array):
    return np.min(array)

def get_min_index(array):
    min_index = np.argmin(array)
    min  = array[min_index]
    return min, min_index

def Euclidian_distance(coor_1, coor_2):
    x1, y1 = coor_1
    x2, y2 = coor_2
    return int(((x2 -x1)**2 + (y2-y1)**2 )**(1/2))


def Random(cost_list, distance_matrix, lista):
    lenght = len(lista)
    total_cost = 0
    edges = []
    for i in range(lenght):
        a = lista[i]
        b = lista[(i+1)%lenght]
        dist = distance_matrix[a][b]
        cost = cost_list[i]
        total_cost += dist + cost 
        edges.append([a,b])
    return edges, total_cost


def get_dist_matrix_and_cost(data, cost = True):
    NR_NODES = len(data)
    distance_matrix = np.zeros((NR_NODES,NR_NODES))
    cost_list = np.zeros(NR_NODES)
    for i in range(NR_NODES):
        for j in range(i,NR_NODES):
            dist = Euclidian_distance(data[i][:2],data[j][:2])
            if cost:
                cost_list[i] = data[i][2]
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
            distance_matrix[i][i] = np.inf
        
    return distance_matrix, cost_list


def add_to_cycle(edge, new_n, dist_m,edges, cost_list):
    #
    start_n = edge[0]
    end_n = edge[1]
    # cost(start-end) + cost(start, new) + cost(end,new) + cost(new)

    cost_diff = -dist_m[start_n][ end_n] + dist_m[start_n][ new_n] + dist_m[new_n][end_n] + cost_list[new_n]
    edges.remove(edge)
    edges.append([start_n, new_n])
    edges.append([new_n, end_n])
    return edges, cost_diff

def cound_cost_diff_cycle(edge, new_n, dist_m, cost_list):
    start_n = edge[0]
    end_n = edge[1]
    # cost(start-end) + cost(start, new) + cost(end,new) + cost(new)print(distance_matrix[:10][:10])
    cost_diff = -dist_m[start_n][ end_n] + dist_m[start_n][ new_n] + dist_m[new_n][end_n] + cost_list[new_n]

    return cost_diff

def repeat(method, indices, distance_matrix, cost_list, NR_NODES, HALF_NODES):
    total_cost = []
    best_cost = np.inf
    best_sol = -1
    best_ind = -1
    for i in indices:
        cost, sol = method(distance_matrix, cost_list, NR_NODES, HALF_NODES, i)
        total_cost.append(cost)
        if cost<best_cost:
            best_cost = cost
            best_sol = sol
            best_ind = i
    return total_cost, best_sol, best_ind

def present_statistic(list):
    res = return_statistic(list)
    print("MIN: ",res[0])
    print("MAX: ",res[1])
    print("AVG: ",res[2])
    print("STD: ",res[3])

def return_statistic(list):
    return min(list), max(list), np.mean(list),  np.std(list)

def count_cost_diff_cycle(edge, new_n, dist_m, cost_list):
    start_n = edge[0]
    end_n = edge[1]
    # cost(start-end) + cost(start, new) + cost(end,new) + cost(new)print(distance_matrix[:10][:10])
    cost_diff = - dist_m[start_n][ end_n] + dist_m[start_n][ new_n] + dist_m[new_n][end_n] +cost_list[new_n]
    # assert cost_diff>0, f'{cost_diff, - dist_m[start_n][ end_n], dist_m[start_n][ new_n], dist_m[new_n][end_n]}' #
    return cost_diff

def create_regret_matrix(non_visited, cur_tour, dist_m, cost_list): #cur_tour = edges
    reg_matrix = np.zeros((len(dist_m),len(cur_tour)))
    for new_node in non_visited:
        for i,edge in enumerate(cur_tour):
            reg_matrix[new_node][i]=count_cost_diff_cycle(edge, new_node, dist_m, cost_list)

    return reg_matrix

def return_biggest_regret(matrix):
    min_values_for_rows = np.min(matrix, axis=1)
    # print("min_values_for_rows",min_values_for_rows)
    rescue_node = np.argmax(min_values_for_rows, axis=0) #najlepiej ratować 4 index
    # print("City with bigest regret",rescue_node)
    rescueing_node = np.argmin(matrix[rescue_node])# def return_max_from_min_rows_regret(matrix):
    # print("Rescueing edge index to modify: ",rescueing_node)
    return rescue_node, rescueing_node

def get_data(path):
    
    with open(path, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=';'))
        for item in range(len(data)):
            i = data[item]
            data[item] = [int(i[0]),int(i[1]),int(i[2])]
    return data
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
from utils import *

def two_regret_greedy_cycle(distance_matrix,cost_list,NR_NODES, HALF_NODES, starting_node, verbose = False):
    a= starting_node
    b = np.argmin(distance_matrix[a]+ cost_list)
    if verbose:
        print(a,b)
    cur_tour = [(a,b),(b,a)]
    non_visited = list(x for x in range(NR_NODES))
    non_visited.remove(cur_tour[0][0])
    non_visited.remove(cur_tour[0][1])
    if verbose:
        print(non_visited, cur_tour)
    total_cost = cost_list[cur_tour[0][1]] + cost_list[cur_tour[0][0]] + 2 * distance_matrix[cur_tour[0][1]][cur_tour[0][0]]

    for i in range(HALF_NODES):
        start_dist_matrix = create_regret_matrix(non_visited, cur_tour, distance_matrix, cost_list)
        start_dist_for_nodes = np.min(start_dist_matrix, axis = 1)

        
        max_regrets = [-1 for i in range(NR_NODES)]
        for new_node in non_visited:
            #we assume that we are adding new node in the shortest way
            edge_index = np.argmin(start_dist_matrix[new_node], axis=0)
            edge = cur_tour[edge_index]

            #if we add new_node in place of edge
            sim_non_visited = deepcopy(non_visited)
            sim_cur_tour = deepcopy(cur_tour)
            sim_cur_tour, cost_diff = add_to_cycle(edge, new_node, distance_matrix, sim_cur_tour, cost_list)
            sim_non_visited.remove(new_node)
            
            
            sim_regret_matrix = create_regret_matrix(sim_non_visited, [(new_node, edge[0]), (edge[1], new_node)] , distance_matrix, cost_list) #[(new_node, edge[0]), (edge[1], new_node)]
            #create sth that copy the start dist-matrix but without edge and add sim_m_regret matrix to count the min 
            
            #min distance to every node after we insert node in particular place,
            new_edges_dist = np.min(sim_regret_matrix, axis = 1)

            if len(cur_tour)==2: #only two edges
                 distances_after_inserting = new_edges_dist

            if edge_index == 0: #first edge
                reused_dist_2 = np.min(start_dist_matrix[:,edge_index+1:], axis = 1)
                distances_after_inserting = np.minimum(reused_dist_2, new_edges_dist) 

            elif edge_index == len(cur_tour)-1: #last edge
                reused_dist_1 = np.min(start_dist_matrix[:,: edge_index], axis = 1)
                distances_after_inserting = np.minimum(reused_dist_1, new_edges_dist) 

            else:
                reused_dist_1 = np.min(start_dist_matrix[:,: edge_index], axis = 1)
                reused_dist_2 = np.min(start_dist_matrix[:,edge_index+1:], axis = 1)
                distances_after_inserting = np.minimum(reused_dist_1,reused_dist_2, new_edges_dist) 


            regret = distances_after_inserting - start_dist_for_nodes

            # print(pd.DataFrame(zip(start_dist_for_nodes,distances_after_inserting, regret)))
            max_regrets[new_node] = np.max(regret)
        added_node = np.argmax(max_regrets)
        edge = cur_tour[np.argmin(start_dist_matrix[added_node], axis=0)]
        
        cur_tour, cost_diff = add_to_cycle(edge, added_node, distance_matrix, cur_tour, cost_list)
        non_visited.remove(added_node)
        total_cost += cost_diff
        if verbose:
            print(i)
            print("COST", total_cost)
            print("ADDED", added_node)
            print("CUR_TOUR",cur_tour)
    print(starting_node, total_cost)
    return total_cost, cur_tour



def greedy_cycle(distance_matrix,cost_list,NR_NODES, HALF_NODES, starting_node ):

    min, second_node = get_min_index(distance_matrix[starting_node]+cost_list[starting_node])

    non_visited = list(range(NR_NODES))
    non_visited.remove(starting_node)
    non_visited.remove(second_node)

    edges = [[starting_node, second_node],[second_node, starting_node]]
    cost = cost_list[starting_node] + cost_list[second_node] + 2 * distance_matrix[starting_node][second_node]

    for n in range(HALF_NODES-2):
        min_cost_diff = np.inf
        best_min_edge = [-1,-1]
        best_new_node = -1

        for e in edges:
            for new_node in non_visited:
                
                cost_diff = -distance_matrix[e[0]][e[1]] \
                    + distance_matrix[e[0]][ new_node] + distance_matrix[new_node][e[1]] \
                    + cost_list[new_node]

                if cost_diff < min_cost_diff:
                    best_min_edge = e
                    best_new_node = new_node
                    min_cost_diff = cost_diff

        cost += min_cost_diff
        non_visited.remove(best_new_node)

        edges.remove(best_min_edge)
        edges.append([best_min_edge[0], best_new_node])
        edges.append([best_new_node, best_min_edge[1]])
    return cost, edges


def KNN(distance_matrix, cost_list, NR_NODES, HALF_NODES,starting_node):

    visited_ind = [starting_node]

    cost_matrix = distance_matrix + cost_list.T

    min, ind = get_min_index(cost_matrix[starting_node])

    visited_ind.append(ind)

    cost = cost_list[starting_node]+cost_list[ind] + distance_matrix[starting_node, ind]

    cost_matrix[:, visited_ind] =  np.inf

    edges = [[starting_node, ind]]
    for n in range(HALF_NODES-2):
        min_edge = np.inf
        start_min_ind = -1
        end_min_ind = -1

        for vis_ind in visited_ind:
            min, ind = get_min_index(cost_matrix[vis_ind])

            if min < min_edge:
                min_edge = min
                start_min_ind = vis_ind
                end_min_ind = ind

        cost += min_edge

        visited_ind.remove(start_min_ind)
        visited_ind.append(end_min_ind)

        edges.append([start_min_ind, end_min_ind])

        cost_matrix[:, end_min_ind] =  np.inf

    edges.append(visited_ind)
    cost+=distance_matrix[visited_ind[0],visited_ind[1]]
    return cost, edges



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
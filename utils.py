import matplotlib.pyplot as plt
import numpy as np

def plotMap(nodes, edges=[], colors = False):
    fig=plt.figure(figsize=(6,6), dpi= 100, facecolor='w', edgecolor='k')
    
    for e in (edges):
        start, end = e
        X = [nodes[start][0], nodes[end][0]]
        
        Y = [nodes[start][1], nodes[end][1]]
        plt.plot(X, Y, lw=1, ls="-", marker="", color = 'gray' )
        
    X = [c[0] for c in nodes]
    Y = [c[1] for c in nodes]
    S = [c[2] for c in nodes]
    
    # plt.scatter(X, Y, S, c = S, cmap="RdYlGn", )
    # plt.scatter(X, Y, c = S, cmap="RdYlGn", )
    plt.scatter(X, Y, c = S, cmap="grey", )


    
    plt.show()


def get_min_value(array):
    return np.min(array)

def get_min_index(array):
    min = get_min_value(array)
    return min, np.where(array == min)[0][0]

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


def get_dist_matrix_and_cost(data):
    NR_NODES = len(data)
    distance_matrix = np.zeros((NR_NODES,NR_NODES))
    cost_list = np.zeros(NR_NODES)
    for i in range(NR_NODES):
        for j in range(i,NR_NODES):
            dist = Euclidian_distance(data[i][:2],data[j][:2])
            cost_list[i] = data[i][2]
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
            distance_matrix[i][i] = np.inf
        
    return distance_matrix, cost_list

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

def add_to_cycle(edge, new_n, dist_m,edges):
    #
    start_n = edge[0]
    end_n = edge[1]
    # cost(start-end) + cost(start, new) + cost(end,new) + cost(new)

    cost_diff = - dist_m[[start_n, end_n]] + dist_m[[start_n, new_n]] + dist_m[[new_n, end_n]] + dist_m[[new_n, new_n]]
    edges.remove([edge])
    edges.append([start_n, new_n])
    edges.append([new_n, end_n])
    return edges, cost_diff

def cound_cost_diff_cycle(edge, new_n, dist_m, cost_list):
    start_n = edge[0]
    end_n = edge[1]
    # cost(start-end) + cost(start, new) + cost(end,new) + cost(new)print(distance_matrix[:10][:10])
    cost_diff = -dist_m[start_n][ end_n] + dist_m[start_n][ new_n] + dist_m[new_n][end_n] + cost_list[new_n]
    return cost_diff

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
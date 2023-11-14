import numpy as np
import matplotlib.pyplot as plt
import csv

def get_data(path):
    
    with open(path, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=';'))
        for item in range(len(data)):
            i = data[item]
            data[item] = [int(x) for x in i]
    return data

def plotMap(nodes, edges=[], tour = [],colors = False, cost = True):
    fig=plt.figure(figsize=(6,6), dpi= 100, facecolor='w', edgecolor='k')
    if edges ==[] : #[ordered nodes not edges
        for i in range(len(tour)):
            edges.append([tour[i],tour[(i+1)%len(tour)]])
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

def Euclidian_distance(coor_1, coor_2):
    x1, y1 = coor_1
    x2, y2 = coor_2
    return int(((x2 -x1)**2 + (y2-y1)**2 )**(1/2))

def create_dist_matrix_and_cost(data, infinity = False):
        NR_NODES = len(data)
        node_distances = np.zeros((NR_NODES,NR_NODES))
        cost_list = np.zeros(NR_NODES)
        for i in range(NR_NODES):
            for j in range(i,NR_NODES):
                dist = Euclidian_distance(data[i][:2],data[j][:2])
                cost_list[i] = data[i][2]
                node_distances[i][j] = dist
                node_distances[j][i] = dist
                if infinity:
                    node_distances[i][i] = np.inf
        return node_distances, cost_list

def count_cost_diff_cycle(edge, new_n, dist_m, cost_list):
    start_n = edge[0]
    end_n = edge[1]
    # cost(start-end) + cost(start, new) + cost(end,new) + cost(new)print(distance_matrix[:10][:10])
    cost_diff = - dist_m[start_n][ end_n] + dist_m[start_n][ new_n] + dist_m[new_n][end_n] +cost_list[new_n]
    # assert cost_diff>0, f'{cost_diff, - dist_m[start_n][ end_n], dist_m[start_n][ new_n], dist_m[new_n][end_n]}' #
    return cost_diff

def objective_function(solution: list, node_distances:np.array, cost_list: np.array):
    length = len(solution)
    total_cost = 0 
    for i in range(length):
        a = solution[i]
        b = solution[(i+1)%length]
        dist = node_distances[a][b]
        cost = cost_list[i]
        total_cost += dist + cost 
    return total_cost

def count_swap_inter_nodes(ind_1, ind_2, solution: list, node_distances:np.array, cost_list: np.array):
    holder = solution[ind_1] #wartość pierwszego node do swap
    solution[ind_1] = solution[ind_2]
    solution[ind_2] = holder

    total = objective_function(solution, node_distances, cost_list)
    
    solution[ind_2] = solution[ind_1]
    solution[ind_1] = holder
    return total

def count_swap_outer_nodes(ind_1, node_2, solution: list, node_distances:np.array, cost_list: np.array):
    holder = solution[ind_1] #wartość pierwszego node do swap
    solution[ind_1] = node_2

    total = objective_function(solution, node_distances, cost_list)
    
    solution[ind_1] = holder
    return total

def min_swap_inter_nodes(solution, node_distances, cost_list):
    length = len(solution)
    delta_matrix = np.zeros((length,length))
    delta_matrix.fill(np.inf)
    for ind_1 in range(length):
        for  ind_2 in range(ind_1+1, length):

            delta_matrix[ind_1][ind_2]=count_swap_inter_nodes(ind_1,ind_2, solution, node_distances, cost_list)
    #TODO: optimize swap
    index_of_min = np.unravel_index(np.argmin(delta_matrix), delta_matrix.shape)
    minimum_value = delta_matrix[index_of_min]
    return index_of_min,minimum_value

def min_swap_outer_nodes(solution, unvisited, node_distances, cost_list, NR_NODES):
    delta_matrix = np.zeros((len(solution),len(unvisited)))
    delta_matrix.fill(np.inf)
    for ind_1 in range(len(solution)):
        for ind_2 in range(len(unvisited)):
            delta_matrix[ind_1][ind_2]=count_swap_outer_nodes(ind_1,unvisited[ind_2], solution,  node_distances, cost_list)

    #TODO: optimize swap
    index_of_min = np.unravel_index(np.argmin(delta_matrix), delta_matrix.shape)
    minimum_value = delta_matrix[index_of_min]

    return index_of_min, minimum_value

def apply_swap_inter_nodes(ind_1, ind_2, solution):
    holder = solution[ind_1] #wartość pierwszego node do swap
    solution[ind_1] = solution[ind_2]
    solution[ind_2] = holder
    return solution

def apply_swap_outer_nodes(ind_1, ind_2, unvisited, solution):
     #wartość pierwszego node do swap
    holder = unvisited[ind_2] 
    unvisited[ind_2] = solution[ind_1]

    print(solution[ind_1] ," -> ", holder)
    solution[ind_1] = holder
    return solution, unvisited

def get_best_solution_for_D(edges = []):
    if edges == []:
        return [95, 172, 16, 18, 132, 185, 73, 136, 61, 33, 29, 12, 107, 139, 44, 117, 196, 150, 162, 67, 114, 85, 129, 64, 89, 159, 147, 58, 171, 72, 71, 119, 59, 193, 166, 28, 110, 158, 156, 91, 51, 70, 174, 140, 148, 141, 142, 130, 188, 161, 192, 21, 138, 82, 115, 8, 63, 40, 163, 182, 0, 57, 102, 37, 165, 194, 134, 36, 25, 154, 112, 152, 50, 131, 103, 38, 31, 101, 197, 183, 34, 122, 24, 127, 121, 179, 143, 169, 66, 99, 137, 88, 153, 145, 157, 80, 19, 190, 198, 135]
    else:
    # edges = [[95, 172], [172, 16], [16, 18], [18, 132], [132, 185], [185, 73], [73, 136], [95, 135], [135, 198], [198, 190], [190, 19], [136, 61], [19, 80], [80, 157], [157, 145], [145, 153], [153, 88], [88, 137], [137, 99], [99, 66], [66, 169], [169, 143], [143, 179], [179, 121], [121, 127], [127, 24], [24, 122], [122, 34], [34, 183], [183, 197], [197, 101], [101, 31], [31, 38], [38, 103], [103, 131], [131, 50], [61, 33], [33, 29], [29, 12], [12, 107], [107, 139], [139, 44], [44, 117], [117, 196], [196, 150], [150, 162], [162, 67], [67, 114], [114, 85], [85, 129], [129, 64], [64, 89], [89, 159], [159, 147], [147, 58], [58, 171], [171, 72], [72, 71], [71, 119], [119, 59], [59, 193], [193, 166], [166, 28], [50, 152], [152, 112], [112, 154], [154, 25], [25, 36], [36, 134], [134, 194], [194, 165], [165, 37], [37, 102], [28, 110], [110, 158], [158, 156], [156, 91], [91, 51], [51, 70], [70, 174], [174, 140], [140, 148], [148, 141], [141, 142], [142, 130], [130, 188], [188, 161], [102, 57], [57, 0], [161, 192], [192, 21], [21, 138], [138, 82], [82, 115], [115, 8], [8, 63], [63, 40], [40, 163], [163, 182], [0, 182]]
        l = edges.pop(0)
        print(len(edges))
        length = len(edges)
        for i in range(length):
            # print(l)
            for j in range(len(edges)):
                e = edges[j]
                if l[-1] == e[0]:
                    if e[1] in l:
                        print("hello",l, e)
                        # break
                    l.append(e[1])
                    edges.pop(j)
                    break
                if l[-1] == e[1]:
                    if e[0] in l:
                        print('hello1',l, e, )
                        break
                    l.append(e[0])
                    edges.pop(j)
                    break
        return l

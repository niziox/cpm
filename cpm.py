#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
import plotly.figure_factory as ff

def numeration(weight_matrix):
    weight_matrix = np.array(weight_matrix)
    num_matrix = np.zeros(weight_matrix.shape)
    weight_num_matrix = deepcopy(weight_matrix)
    next_number = 0
    was_zero = True
    transform = {}

    while was_zero:
        list_remove = [-1] * weight_matrix.shape[1]

        for row in range(weight_matrix.shape[0]):
            for col in range(weight_matrix.shape[1]):
                if weight_matrix[row, col] != -1:
                    if list_remove[col] == -1:
                        list_remove[col] = 0
                    list_remove[col] += weight_matrix[row, col]

        was_zero = False
        for idx, elem_remove in enumerate(list_remove):
            if elem_remove == 0:
                transform[idx+1] = next_number+1

                weight_matrix[:, idx] = -1
                weight_matrix[idx, :] = -1

                next_number += 1
                was_zero = True

    for row in range(weight_num_matrix.shape[0]):
        for col in range(weight_num_matrix.shape[1]):
            num_matrix[transform[row+1]-1, transform[col+1]-1] = weight_num_matrix[row, col]

    return num_matrix, transform

def time_calculate(matrix_graph):
    # zamiana macierzy na liste sąsiedztwa
    dict_graph = {}
    for a, b in enumerate(matrix_graph, 1):
        dict_graph[a] = []
        for c, d in enumerate(b, 1):
            for e in range(int(d)):
                if c not in dict_graph[a]:
                    dict_graph[a].append(c)
        if not dict_graph[a]:
            del dict_graph[a]

    nw_np_dict = {node : [0, np.inf] for node in dict_graph.keys()}
    nw_np_dict[list(dict_graph.keys())[-1] + 1] = [0, np.inf]

    # najwcześniejszy możliwy termin
    for i in dict_graph.keys():
        for j in dict_graph[i]:
            t_ij = nw_np_dict[i][0] + matrix_graph[i-1, j-1]
            prev = nw_np_dict[j][0]
            nw_np_dict[j][0] = max(prev, t_ij)

    nw_np_dict[list(dict_graph.keys())[-1] + 1][1] = nw_np_dict[list(dict_graph.keys())[-1] + 1][0]

    # najpóźniejszy możliwy termin
    for i in list(nw_np_dict.keys())[-2:0:-1]:
        for j in dict_graph[i][::-1]:
            t_ij = nw_np_dict[j][1] - matrix_graph[i-1, j-1]
            prev = nw_np_dict[i][1]
            nw_np_dict[i][1] = min(prev, t_ij)

    nw_np_dict[list(dict_graph.keys())[0]][1] = nw_np_dict[list(dict_graph.keys())[0]][0]

    return nw_np_dict


def critical_path(matrix_graph, time_dict):
    # zamiana macierzy na liste sąsiedztwa
    dict_graph = {}
    for a, b in enumerate(matrix_graph, 1):
        dict_graph[a] = []
        for c, d in enumerate(b, 1):
            for e in range(int(d)):
                if c not in dict_graph[a]:
                    dict_graph[a].append(c)
        if not dict_graph[a]:
            del dict_graph[a]

    # zapas całkowity
    f_float = {}

    for i in dict_graph.keys():
        for j in dict_graph[i]:
            f_float[(i, j)] = time_dict[j][1] - time_dict[i][0] - matrix_graph[i-1, j-1]

    # ścieżka krytyczna
    critical_path_list = [1]

    for path in f_float.keys():
        if f_float[path] == 0:
            critical_path_list.append(path[1])

    # zapas swobodny
    s_float = {}

    for i in dict_graph.keys():
        for j in dict_graph[i]:
            s_float[(i, j)] = time_dict[j][0] - time_dict[i][0] - matrix_graph[i-1, j-1]

    # zapas niezależny
    n_float = {}

    for i in dict_graph.keys():
        for j in dict_graph[i]:
            n_float[(i, j)] = max(0, time_dict[j][0] - time_dict[i][1] - matrix_graph[i-1, j-1])

    # harmonogram realizacji przedsięwzięcia
    timetable = {}

    for i in dict_graph.keys():
        for j in dict_graph[i]:
            tij = matrix_graph[i-1, j-1]
            timetable[(i, j)] = [time_dict[i][0], time_dict[j][1] - tij, time_dict[i][0] + tij, time_dict[j][1]]

    return critical_path_list, f_float, s_float, n_float, timetable

def cpm(matrix_graph):

    new_numeration = numeration(m)

    times = time_calculate(new_numeration[0])

    critical_path_cpm, f_float, s_float, n_float, timetable = critical_path(new_numeration[0], times)

    return times, critical_path_cpm, f_float, s_float, n_float, timetable


if __name__ == '__main__':
    m = [[0, 2, 3, 0, 0, 0, 0, 6, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 5, 4, 0, 0, 0],
         [0, 0, 0, 7, 0, 0, 2, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 5, 4, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
         [0, 0, 0, 1, 0, 5, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 2, 0, 1, 3, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
         [0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 8, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    times, critical_path_cpm, f_float, s_float, n_float, timetable = cpm(m)

    print('Numer zdarzenia : [najwcześniejszy możliwy termin, najpóźniejszy możliwy termin]')
    print(times)

    print('\n\nKolejne zdarzenia ścieżki krytycznej')
    print(critical_path_cpm)

    print('\n\nCzynność : zapas całkowity')
    print(f_float)

    print('\n\nCzynność : zapas swobodny')
    print(s_float)

    print('\n\nCzynność : zapas niezależny')
    print(n_float)

    print('\n\nHarmonogram realizacji przedsięwzięcia')
    for table in timetable.keys():
        timetable[table] += [f_float[table], s_float[table], n_float[table]]
    print(timetable)

    # Harmonogram Gantt'a
    df = []
    for act in timetable.keys():
        if timetable[act][4] == 0 and timetable[act][5] == 0 and timetable[act][5] == 0:
            df.append((dict(Task=act, Start=f'2021-05-{int(timetable[act][0])}', Finish=f'2021-05-{int(timetable[act][3])}', Resource='Ścieżka krytyczna')))
        else:
            df.append((dict(Task=act, Start=f'2021-05-{int(timetable[act][0])}', Finish=f'2021-05-{int(timetable[act][1])}', Resource='Zakres trwania')))
            df.append((dict(Task=act, Start=f'2021-05-{int(timetable[act][2])}', Finish=f'2021-05-{int(timetable[act][3])}', Resource='Zakres ukończenia')))

    colors = {'Ścieżka krytyczna': 'rgb(220, 0, 0)', 'Zakres trwania': (1, 0.9, 0.16), 'Zakres ukończenia': 'rgb(0, 255, 100)'}

    fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True, group_tasks=True)
    fig.show()

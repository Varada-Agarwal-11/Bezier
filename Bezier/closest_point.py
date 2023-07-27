import numpy as np
from iterative_method_fit import closest_point
import plotly.graph_objects as go
import csv
from common_funcs import get_random_color
import plotly.figure_factory as ff


def bezier(a, b, c, d, t):
    if t is None:
        # Handle the case when t is None
        return None, None
    return (
        np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d,
        -3 * np.power(1 - t, 2) * a + 3 * np.power(1 - t, 2) * b - 6 * (1 - t) * t * b + 6 * (1 - t) * t * c - 3 * np.power(t, 2) * c + 3 * np.power(t, 2) * d
    )


control_points = []
with open('control_points (1).txt', 'r') as cp_file:
    for line in cp_file:
        points = [float(val) for val in line.strip().split(',')]
        control_points.append([(points[i], points[i + 1]) for i in range(0, len(points), 2)])
# print(control_points)

noisy_points = []
with open('noisy_points_25.csv', 'r') as np_file:
    csv_reader = csv.reader(np_file)
    for row in csv_reader:
        x, y = [float(coord) for coord in row]
        noisy_points.append((x, y))
# print(len(noisy_points))

path = []
with open('path.csv', 'r') as np_file:
    csv_reader = csv.reader(np_file)
    for row in csv_reader:
        x, y = [float(coord) for coord in row]
        path.append((x, y))

ppl_projected =[]
tangent = []
with open('data_25_projected.csv', 'r') as np_file:
    csv_reader = csv.reader(np_file)
    next(csv_reader)
    for row in csv_reader:
        x, y = [float(coord) for coord in row[2:4]]
        x_, y_ = [float(coord) for coord in row[7:9]]
        ppl_projected.append((x, y))
        tangent.append((x_, y_))
print(len(ppl_projected))

results = []
clo_point = []
# tangent = []
for point in noisy_points:
    min_dist = float('inf')
    closest_p = None
    best_t = None
    for i in range(len(control_points) // 3 - 1):
        cp = np.zeros((4, 2))
        cp[0] = control_points[3 * i][0]
        cp[1] = control_points[3 * i + 1][0]
        cp[2] = control_points[3 * i + 2][0]
        cp[3] = control_points[3 * i + 3][0]
        c_state = closest_point(cp, point)
        if c_state["dist"] < min_dist:
            min_dist = c_state["dist"]
            best_t = c_state["parameter"]
            closest_p = bezier(cp[0],cp[1],cp[2],cp[3], best_t)
    clo_point.append(closest_p[0])
#     tangent.append(closest_p[1])
print(len(clo_point))
fig = go.Figure()

ppl_projected  = np.array(ppl_projected)
noisy_points = np.array(noisy_points)
clo_point = np.array(clo_point)
path = np.array(path)

fig = ff.create_quiver(
    ppl_projected[:, 0], ppl_projected[:, 1],  # x, y coordinates of starting points
    noisy_points[:, 0]- ppl_projected[:, 0] , noisy_points[:, 1] - ppl_projected[:, 1],  # u, v coordinates of the vectors (normals)
    scale=1,  # The overall scale factor of the vectors
    arrow_scale=0.2,  # The scale factor of the arrowhead compared to the vector length
    name='normals',  # Name for the quiver trace in the legend
    line_width=1  # Width of the lines used to represent the vectors
)

fig.add_trace(go.Scatter(x=clo_point[:, 0], y=clo_point[:, 1], mode='markers',
                         marker=dict(color=get_random_color(), size=8), name='diff method projected point'))
fig.add_trace(go.Scatter(x=noisy_points[:, 0], y=noisy_points[:, 1], mode='markers',
                         marker=dict(size=4, color=get_random_color()), name='Noisy Points'))
fig.add_trace(go.Scatter(x=[point[0] for point in ppl_projected], y=[point[1] for point in ppl_projected], mode='markers',
                         marker=dict(size=4, color=get_random_color()), name='ppl projected'))
fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode='lines',
                         line=dict(color=get_random_color(), width=2), name='Path'))

fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
fig.show()
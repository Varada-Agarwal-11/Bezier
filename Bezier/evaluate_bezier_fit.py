import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# np.random.seed(42)

def line_2_point(line):
    string_point = line.split('\n')
    x, y = map(float, line.strip().split(','))
    return x,y


def load_points(filepath):
    file = open(filepath, 'r')
    lines = file.readlines()
    points = [line_2_point(x) for x in lines]
    return np.asarray(points)


# find the a & b points
def get_bezier_coef(points):
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = 2 * (2 * points[:-1] + points[1:])
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = np.zeros((n, 2))
    B[:-1] = 2 * points[1:-1] - A[1:]
    B[n - 1] = (A[n - 1] + points[n]) / 2
    return A, B

# returns the general Bezier cubic formula given 4 control points
def get_cubic(a, b, c, d, t):
    return np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

# return one cubic curve for each consecutive points
def get_bezier_cubic(points, t):
    A, B = get_bezier_coef(points)
    broadcast = np.ones_like(t)
    pts_broadcast = points[None, :-1, :] * broadcast
    pts_p1_broadcast = points[None, 1:, :] * broadcast
    return get_cubic(pts_broadcast, A, B, pts_p1_broadcast, t).reshape((-1, 2), order='F')

# evalute each cubic curve on the range [0, 1] sliced in n points
def evaluate_bezier(points, n):
    t = np.linspace(0, 1, n)[:, None, None]
    curves = get_bezier_cubic(points, t)
    return curves


def display_points(points, path):
    layout = go.Layout(scene=dict(aspectmode='data'))
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers',
                             marker=dict(size=10, color='rgb(200, 0, 0)'), name='points'))
    fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode='markers',
                             marker=dict(size=2, color='rgb(0, 0, 200)'), name='bezier_path_fit'))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()

# generate 5 (or any number that you want) random points that we want to fit (or set them youreself)
# points = np.random.rand(5, 2)
file_name = 'noisy_points.txt'
points_ = load_points(file_name)
points = []
for i in range(0, len(points_), 50):
    points.append(points_[i])
points.append(points[0])
points = np.asarray(points)
t = len(points)
path = evaluate_bezier(points[0:t], 100)
display_points(points, path)
import numpy as np
from iterative_method_fit import bezier_curve, get_knot_points, get_bezier_control_points, closest_point, bezier
import plotly.graph_objects as go
from function_discrepancy import discrepancy_test
from common_funcs import get_random_color

t = np.linspace(0, 1, 100)
a, b, c, d = [1.0, 1.0], [2.0, 6.0], [5.0, -5.0], [6.0, 2.0]
ideal_control_pts = np.asarray([[1.0, 1.0], [2.0, 6.0], [5.0, -5.0], [6.0, 2.0]])
x_vals, y_vals = bezier_curve(a, b, c, d, t)
points = np.zeros((100, 2), dtype=float)  # Change dtype to float
points[:, 0] = x_vals
points[:, 1] = y_vals
knot_points, strips = get_knot_points(points)
control_points = get_bezier_control_points(points, strips)

def plot_bez(i):
    new_x_vals, new_y_vals = bezier_curve(control_points[4*i+0], control_points[4*i+1], control_points[4*i+2], control_points[4*i+3], t)
    err = discrepancy_test(control_points[4*i:4*(i+1)], points[strips[i]:strips[i+1]+1])
    print('error', err)

    fig = go.Figure()

    # Plot the points, control points, and new points
    fig.add_trace(go.Scatter(x=points[strips[i]:strips[i+1]+1, 0], y=points[strips[i]:strips[i+1]+1, 1], mode='markers',
                                marker=dict(size=4, color=get_random_color()), name='points'))

    fig.add_trace(go.Scatter(x=control_points[4*i:4*(i+1), 0], y=control_points[4*i:4*(i+1), 1], mode='markers',
                                marker=dict(size=6, color=get_random_color()), name='control_points_strip1'))

    fig.add_trace(go.Scatter(x=new_x_vals, y=new_y_vals, mode='markers',
                                marker=dict(size=6, color=get_random_color()), name='new_points'))

    dashed_lines = []
    for point in points[strips[i]:strips[i + 1] + 1]:
        closest = closest_point(control_points[4 * i:4 * (i + 1)], point)
        closest_x, closest_y = bezier_curve(control_points[4 * i + 0], control_points[4 * i + 1], control_points[4 * i + 2],
                                      control_points[4 * i + 3], closest['parameter'])
        line = go.Scatter(x= [point[0], closest_x],
                          y= [point[1], closest_y],
                          mode='lines',
                          line=dict(dash='dash', color=get_random_color(), width=2),
                          name=f'Dashed Line {i + 1}')
        dashed_lines.append(line)

    fig.add_traces(dashed_lines)

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()


for i in range(3):
    plot_bez(i)
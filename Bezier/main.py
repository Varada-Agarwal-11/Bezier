from function_discrepancy import discrepancy_test
from iterative_method_fit import load_points, get_knot_points, get_bezier_control_points
import numpy as np
import plotly.graph_objs as go
import random
import numpy as np
import csv

def get_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return 'rgb(' + str(r) + ',' + str(g) + ',' + str(b) + ')'
    pass

file_name = 'path.csv'
points_ = load_points(file_name)
random_points = random.sample(range(len(points_)), 25)
print(len(points_))
noisy_points = []
for i in random_points:
    x, y = points_[i]
    x_noise = np.random.normal(0, 0.1)
    y_noise = np.random.normal(0, 0.1)
    noisy_x = x + x_noise
    noisy_y = y + y_noise
    noisy_points.append((noisy_x, noisy_y))

with open('noisy_points_25.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['x', 'y'])  # Write header
    for x, y in noisy_points:
        csv_writer.writerow([x, y])
# knot_points, strips = get_knot_points(points_)
#
# control_points = get_bezier_control_points(points_, strips)
#
# err= discrepancy_test(control_points, points_)
# print('error', err)

# Separate the noisy points into x and y coordinate lists
noisy_x_coords, noisy_y_coords = zip(*noisy_points)



# fig = go.Figure()
# fig.add_trace(go.Scatter(x=[point[0] for point in points_], y=[point[1] for point in points_],
#                          mode='markers', marker=dict(size=2, color='blue'), name='points'))
#
# fig.add_trace(go.Scatter(x=noisy_x_coords, y=noisy_y_coords,
#                          mode='markers', marker=dict(size=4, color='red'), name='noisy points'))
# # fig.add_trace(go.Scatter(x=knot_points[:, 0], y=knot_points[:, 1], mode='markers',
# #                              marker=dict(size=4, color=get_random_color()), name='knot_points'))
# # fig.add_trace(go.Scatter(x=control_points[:, 0], y=control_points[:, 1], mode='markers',
# #                              marker=dict(size=4, color=get_random_color()), name='control_points'))
# fig.update_yaxes(scaleanchor="x", scaleratio=1)
# fig.show()



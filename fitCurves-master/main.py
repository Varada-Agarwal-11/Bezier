from numpy import *
import plotly.graph_objects as go
from bezier import *
from fitCurves import *
import csv

from fitCurves import _winnow

points = []
with open('noisy_points.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        x, y = map(float, row)
        points.append((x, y))

max_error = 0.0001
t = linspace(0, 1, 100)

strips = []
_winnow(points, strips)

ctpt_set = []
for i in range(len(strips) - 1):
    start_idx = strips[i]
    end_idx = strips[i + 1] + 1
    ctpt_set.extend(fitCurve(array(points[start_idx:end_idx]), max_error))
print("No of curves fit  = ", len(ctpt_set))
# print(ctpt_set)

path = []
for ctpt in ctpt_set:
    x, y = zip(*[q(ctpt, ti) for ti in t])
    path.extend(zip(x, y))
x_path, y_path = zip(*path)

fig = go.Figure()

# Plot the path
fig.add_trace(go.Scatter(x=x_path, y=y_path, mode='markers', name='Path'))
fig.add_trace(go.Scatter(x=[point[0] for point in points], y=[point[1] for point in points], mode='markers', name='Input Points'))
fig.update_layout(title='Bezier Curve Fitting',
                  xaxis_title='X-axis',
                  yaxis_title='Y-axis',
                  showlegend=True)

fig.show()
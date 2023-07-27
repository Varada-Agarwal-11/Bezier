#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
from scipy.optimize import fsolve
import plotly.graph_objects as go

def bezier(a, b, c, d, t):
    return (
        np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d,
        -3 * np.power(1 - t, 2) * a + 3 * np.power(1 - t, 2) * b - 6 * (1 - t) * t * b + 6 * (1 - t) * t * c - 3 * np.power(t, 2) * c + 3 * np.power(t, 2) * d
    )
# lower_bound = 0.0
# upper_bound = 1.0

def find_best_t(point, control_points):
    ax, ay = point
    a, a1 = control_points[0]
    b, b1 = control_points[1]
    c, c1 = control_points[2]
    d, d1 = control_points[3]

    x1 = lambda t: bezier(a, b, c, d, t)[0]
    Dx = lambda t: bezier(a, b, c, d, t)[1]
#     roots_Dx = fsolve(Dx, (0.0, 1.0))
    y1 = lambda t: bezier(a1, b1, c1, d1, t)[0]
    Dy = lambda t: bezier(a1, b1, c1, d1, t)[1]
#     roots_Dy = fsolve(Dy, (0.0, 1.0))

    eq1 = lambda t: (ax - x1(t))**2 + (ay - y1(t))**2
    eq2 = lambda t: (ax - x1(t))**2 + np.power(Dy(t), 2)
    eq3 = lambda t: (ay - y1(t))**2 + np.power(Dx(t), 2)
    eq4 = lambda t: (ax - x1(t)) * Dx(t) + (ay - y1(t)) * Dy(t)

    # Define functions that return the equations as a list
    equations = [eq1, eq2, eq3, eq4]
#     equations = [eq1, eq4]

    # Initialize variables to store the minimum distance and corresponding t value
    min_distance = float('inf')
    best_t = None

    # Use fsolve to find the solutions in the range (0, 1)
    for eq in equations:
        roots = fsolve(eq, (0.0, 1.0))
        for root in roots:
            if 0 <= root <= 1:
                distance = np.sqrt((ax - x1(root))**2 + (ay - y1(root))**2)
                if distance < min_distance:
                    min_distance = distance
                    best_t = root

    print("Best t:", best_t)
    return best_t

points = [(3,4), (5,5),(4,5),(1,3)]
controlpoints = [[1.0, 2.0], [3.0, 7.0], [6.0, 2.0], [7.0, 8.0]]

x_coordinates = [point[0] for point in points]
print(x_coordinates)


# In[11]:


best_t = []
best_t2 = [0.281596,0.606079,0.451567, 0.060069]
for point in points:
    t_value = find_best_t(point, controlpoints)
    if t_value is not None:
        best_t.append(t_value)
print(best_t)
# [0.2815964760678439, 0.6061755003547448, 0.4515673160190737, 0.06012506012652572]


# In[3]:



# Generate the Bezier curve points
curve_points = lambda t, control_points: [np.power(1 - t, 3) * control_points[0][0] + 3 * np.power(1 - t, 2) * t * control_points[1][0]                + 3 * (1 - t) * np.power(t, 2) * control_points[2][0] + np.power(t, 3) * control_points[3][0],                np.power(1 - t, 3) * control_points[0][1] + 3 * np.power(1 - t, 2) * t * control_points[1][1]                + 3 * (1 - t) * np.power(t, 2) * control_points[2][1] + np.power(t, 3) * control_points[3][1]]

t = np.linspace(0, 1, 100)
x_coords, y_coords = curve_points(t, controlpoints)
best_fit_x, best_fit_y = curve_points(np.array(best_t), controlpoints)
best_fit_x2, best_fit_y2 = curve_points(np.array(best_t2), controlpoints)

print(best_fit_x2, best_fit_y2 )


# In[4]:


# Create a scatter plot with the control points
scatter = go.Scatter(x=[point[0] for point in controlpoints],
                     y=[point[1] for point in controlpoints],
                     mode='markers',
                     name='Control Points')
scatter1 = go.Scatter(x=[point[0] for point in points],
                      y=[point[1] for point in points],
                      mode='markers',
                      name='points')
scatter2 = go.Scatter(x=[x for x in best_fit_x],
                      y=[y for y in best_fit_y],
                      mode='markers',
                      name='projected_points')

# Create a line plot for the Bezier curve
line2 = go.Scatter(x=x_coords,
                  y=y_coords,
                  mode='lines',
                  name='Cubic Bezier Curve')

# Given best_fit_x and best_fit_y arrays
best_fit_x = np.array([2.86047967, 5.07118395, 4.04490163, 1.37094337])
best_fit_y = np.array([4.31397224, 4.74667725, 4.58980827, 2.79798964])

best_fit_x2 = np.array([2.86047635, 5.07057308, 4.04489945, 1.37058862])
best_fit_y2 = np.array([4.31397076, 4.74650568, 4.58980803, 2.79733812])

# Create a scatter plot for the projected points and points
scatter_points = go.Scatter(x=best_fit_x2,
                            y=best_fit_y2,
                            mode='markers',
                            name=' Actual Projected Points')

# Create a line plot for the dashed lines
dashed_lines = []
for i, point in enumerate(points):
    x_c, y_c = point
    line = go.Scatter(
        x=[x_c, best_fit_x[i]],
        y=[y_c, best_fit_y[i]],
        mode='lines',
        line=dict(dash='dash'),
        name=f'Dashed Line {i+1}'
    )
    dashed_lines.append(line)


# Create the figure and add the scatter and line plots
fig = go.Figure(data=[line2, scatter,scatter1,  *dashed_lines, scatter_points,  scatter2])


# Set layout properties
fig.update_layout(title='Cubic Bezier Curve',
                  xaxis_title='X',
                  yaxis_title='Y')


fig.show()


# In[ ]:





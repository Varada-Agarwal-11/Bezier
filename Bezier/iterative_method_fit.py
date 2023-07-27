#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


# finds the knot points

def _winnow(data, strips):
    if len(data) <= 4:
        raise ValueError("Not enough data to split!")

    strips.clear()
    strips.append(0)

    dir_ch = False
    for i in range(1, len(data) - 1):
        if ((data[i][0] - data[i - 1][0]) * (data[i][0] - data[i + 1][0]) > 0) or (
                (data[i][1] - data[i - 1][1]) * (data[i][1] - data[i + 1][1]) > 0):
            dir_ch ^= True

        if dir_ch:
            strips.append(i)
            dir_ch ^= True

    if strips[-1] != len(data) - 1:
        strips.append(len(data) - 1)
    return strips


def get_knot_points(points_):
    strips = []
    _winnow(points_, strips)
    knot_points = []
    for index in strips:
        knot_points.append(points_[index])

    return np.asarray(knot_points), strips



def line_2_point(line):
    string_point = line.split('\n')
    x, y = map(float, line.strip().split(','))
    return x, y


def load_points(filepath):
    file = open(filepath, 'r')
    lines = file.readlines()
    points = [line_2_point(x) for x in lines]
    return np.asarray(points)




# strip stores index of knot points


# In[33]:





# closest_point returns parameter and distance of closest point on bezier curve (given control points) for a certain data point outside the curve.

def bezier(a, b, c, d, t):
    return (
        np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d,
        -3 * np.power(1 - t, 2) * a + 3 * np.power(1 - t, 2) * b - 6 * (1 - t) * t * b + 6 * (
                    1 - t) * t * c - 3 * np.power(t, 2) * c + 3 * np.power(t, 2) * d
    )


def closest_point(control_points, point):
    ax, ay = point[0], point[1]
    a, a1 = control_points[0][0], control_points[0][1]
    b, b1 = control_points[1][0], control_points[1][1]
    c, c1 = control_points[2][0], control_points[2][1]
    d, d1 = control_points[3][0], control_points[3][1]

    def x1(t):
        return bezier(a, b, c, d, t)[0]
    def Dx(t):
        return bezier(a, b, c, d, t)[1]
    def y1(t):
        return bezier(a1, b1, c1, d1, t)[0]
    def Dy(t):
        return bezier(a1, b1, c1, d1, t)[1]

    def eq1(t):
        return (ax - x1(t)) ** 2 + (ay - y1(t)) ** 2
    def eq2(t):
        return (ax - x1(t)) ** 2 + np.power(Dy(t), 2)
    def eq3(t):
        return (ay - y1(t)) ** 2 + np.power(Dx(t), 2)
    def eq4(t):
        return (ax - x1(t)) * Dx(t) + (ay - y1(t)) * Dy(t)


    equations = [eq1, eq2, eq3, eq4]
    min_distance = float('inf')
    best_t = None

    for eq in equations:
        roots = fsolve(eq, (0.0, 1.0))
        for root in roots:
            if 0 <= root <= 1:
                distance = np.sqrt((ax - x1(root)) ** 2 + (ay - y1(root)) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    best_t = root
    if best_t == None:
        t = np.linspace(0, 1, 100)
        eq1_values = eq1(t)
        min_index = np.argmin(eq1_values)
        best_t = t[min_index]
    C_STATE = {"parameter": best_t, "dist": eq1(best_t)}

    return C_STATE


def discrepancy_test(control_points, points_):
    total_distance = 0.0

    for point in points_:
        closest = closest_point(control_points, point)
        total_distance += closest["dist"]

    return total_distance


# In[34]:


class Vertex:
    # basic functions:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vertex(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vertex(self.x - other.x, self.y - other.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def dist(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


def normalize(vec):
    return vec / np.linalg.norm(vec)


# finds initial parameter value of each point on bezier curve depending on their distance from one another
def centripetal_par(data):
    if len(data) < 2:
        raise ValueError("No data to generate parameter values!")

    para = np.zeros(len(data))
    for i in range(1, len(data)):
        para[i] = para[i - 1] + np.linalg.norm(np.array(data[i]) - np.array(data[i - 1]))

    para /= para[-1]
    return para


# value of tangent on the right side of data point
def right_tangent(data, ind):
    data = np.array(data)  # Convert data to NumPy array
    if ind <= 0:
        raise ValueError("Right tangent called on the most left data point!")
    if ind >= len(data):
        raise ValueError("Out of range index of right tangent!")

    if ind == len(data) - 1:
        return normalize(data[-2] - data[-1])
    else:
        return normalize(data[ind - 1] - data[ind + 1])


# value of tangent on the left side of data point
def left_tangent(data, ind):
    data = np.array(data)  # Convert data to NumPy array
    if ind >= len(data) - 1:
        raise ValueError("Left tangent called on the most right data point!")
    if ind < 0:
        raise ValueError("Out of range index of left tangent!")

    if ind == 0:
        return normalize(data[1] - data[0])
    else:
        return normalize(data[ind + 1] - data[ind - 1])


# coefficients of bezier curve
def basisF(t):
    return np.array([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3])


# In[35]:


# code is fine till here.


# In[36]:


# finds control points between two knot points
def attempt_to_fit(data, _sec, TOLEZ, VERGENCE=1e-6, IT_OVR_FLW=100):
    if _sec.len < 4:
        raise ValueError("Not enough data to fit!")

    controlP = [None] * 4
    fitted_CP = [None] * 4
    fitted_CP[0] = controlP[0] = np.array(data[_sec.f])
    fitted_CP[3] = controlP[3] = np.array(data[_sec.l])

    V1 = left_tangent(data, _sec.f)
    V2 = right_tangent(data, _sec.l)
    para = centripetal_par(data[_sec.f:_sec.l + 1])
    path = None
    STATE = {"ERR": 0.0, "IND": 0}
    OUTCOME = None
    CONV = float('inf')
    P_DIST = float('inf')

    k = 0
    while CONV >= VERGENCE and P_DIST >= TOLEZ and k < IT_OVR_FLW:
        C11, C12, C22, X1, X2 = 0.0, 0.0, 0.0, 0.0, 0.0
        for i in range(_sec.len):
            A1 = basisF(para[i])[1] * V1
            A2 = basisF(para[i])[2] * V2

            C11 += np.dot(A1, A1)
            C12 += np.dot(A1, A2)
            C22 += np.dot(A2, A2)

            if i == 0:
                V0 = np.array(
                    [data[_sec.f + i][0] - ((para[i] + para[i + 1]) * controlP[0][0] + para[i + 1] * controlP[3][0]),
                     data[_sec.f + i][1] - ((para[i] + para[i + 1]) * controlP[0][1] + para[i + 1] * controlP[3][1])])
            elif i == _sec.len - 1:
                V0 = np.array(
                    [data[_sec.f + i][0] - (para[i] * controlP[0][0] + (para[i] + para[i - 1]) * controlP[3][0]),
                     data[_sec.f + i][1] - (para[i] * controlP[0][1] + (para[i] + para[i - 1]) * controlP[3][1])])
            else:
                V0 = np.array(
                    [data[_sec.f + i][0] - ((para[i] + para[i + 1]) * controlP[0][0] + para[i + 1] * controlP[3][0]),
                     data[_sec.f + i][1] - ((para[i] + para[i + 1]) * controlP[0][1] + para[i + 1] * controlP[3][1])])

            X1 += np.dot(A1, V0)
            X2 += np.dot(A2, V0)

        det = C11 * C22 - C12 * C12
        alph1 = 0.0 if det == 0.0 else (C22 * X1 - C12 * X2) / det
        alph2 = 0.0 if det == 0.0 else (C11 * X2 - C12 * X1) / det

        if alph1 <= 0.0 or alph2 <= 0.0:
            midpoint = 0.5 * (np.array(fitted_CP[0]) + np.array(fitted_CP[3]))
            alph1 = np.linalg.norm(midpoint - np.array(fitted_CP[0]))

            fitted_CP[1] = alph1 * V1 + np.array(fitted_CP[0])
            fitted_CP[2] = alph1 * V2 + np.array(fitted_CP[3])

        else:
            fitted_CP[1] = alph1 * V1 + fitted_CP[0]
            fitted_CP[2] = alph2 * V2 + fitted_CP[3]

        STATE["ERR"] = 0.0
        for j in range(_sec.f, _sec.l + 1):
            C_STATE = closest_point(fitted_CP, data[j])
            para[j - _sec.f] = C_STATE["parameter"]
            if C_STATE["dist"] > STATE["ERR"]:
                STATE["ERR"] = C_STATE["dist"]
                STATE["IND"] = j

        CONV = P_DIST - STATE["ERR"]
        if STATE["ERR"] < P_DIST:
            controlP[1] = fitted_CP[1]
            controlP[2] = fitted_CP[2]
            OUTCOME = STATE
        P_DIST = STATE["ERR"]
        k += 1

    return controlP


# In[40]:


class Sector:
    # stores first and last point of a series of points between two knot points
    def __init__(self, f, l):
        self.f = f
        self.l = l
        self.len = l - f + 1


# divides data into segaments based on knot points and finds control points
def get_bezier_control_points(points_, strips):
    control_points = []
    for i in range(len(strips) - 1):
        start_idx = strips[i]
        end_idx = strips[i + 1]
        interval_points = points_[start_idx:end_idx + 1].tolist()
        controlP = []
        _sec = Sector(0, len(interval_points) - 1)
        TOLEZ = 0.01

        controlP = attempt_to_fit(interval_points, _sec, TOLEZ)
        #         controlP = attempt_to_fit(interval_points, _sec)
        control_points.extend(controlP)

    #     control_points.append(points_[strips[-1]])

    return np.asarray(control_points)


# In[41]:




# In[42]:




# # Convert the control_points list to a numpy array
# control_points = np.array(control_points)

# # # Evaluate the Bezier curve using evaluate_bezier function (same as before)
# # n = 100  # Number of points to evaluate on each cubic curve
# # bezier_path = evaluate_bezier(control_points, n)

# # # Display the points and the Bezier path
# # display_points(points_, bezier_path)


# In[43]:



# In[53]:





def bezier(a, b, c, d, t):
    return (
        np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d,
        -3 * np.power(1 - t, 2) * a + 3 * np.power(1 - t, 2) * b - 6 * (1 - t) * t * b + 6 * (
                    1 - t) * t * c - 3 * np.power(t, 2) * c + 3 * np.power(t, 2) * d
    )


def plot_bezier_curve(control_points, strips):
    t = np.linspace(0, 1, 100)
    x_vals = []
    y_vals = []

    for i in range(len(strips) - 1):
        x_seg = []
        y_seg = []

        for t_val in t:
            x, y = bezier(*control_points[4 * i:4 * i + 4], t_val)
            x_seg.append(x)
            y_seg.append(y)

        x_vals.extend(x_seg)
        y_vals.extend(y_seg)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='Bezier Curve')
    plt.scatter(*zip(*control_points), color='red', label='Control Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bezier Curve with Control Points')
    plt.legend()
    plt.grid(True)
    plt.show()


# Rest of the code and the bezier function are the same as in the previous responses.



# In[61]:


def bezier_curve(a, b, c, d, t):
    x = np.power(1 - t, 3) * a[0] + 3 * np.power(1 - t, 2) * t * b[0] + 3 * (1 - t) * np.power(t, 2) * c[0] + np.power(
        t, 3) * d[0]
    y = np.power(1 - t, 3) * a[1] + 3 * np.power(1 - t, 2) * t * b[1] + 3 * (1 - t) * np.power(t, 2) * c[1] + np.power(
        t, 3) * d[1]
    return x, y






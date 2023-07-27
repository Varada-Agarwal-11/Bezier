import numpy as np
from scipy.optimize import fsolve

#closest_point returns parameter and distance of closest point on bezier curve (given control points) for a certain data point outside the curve.

def bezier(a, b, c, d, t):
    return (
        np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d,
        -3 * np.power(1 - t, 2) * a + 3 * np.power(1 - t, 2) * b - 6 * (1 - t) * t * b + 6 * (1 - t) * t * c - 3 * np.power(t, 2) * c + 3 * np.power(t, 2) * d
    )

def closest_point(control_points, point):
    ax, ay = point[0], point[1]
    a, a1 = control_points[0][0], control_points[0][1]
    b, b1 = control_points[1][0], control_points[1][1]
    c, c1 = control_points[2][0], control_points[2][1]
    d, d1 = control_points[3][0], control_points[3][1]

    x1 = lambda t: bezier(a, b, c, d, t)[0]
    Dx = lambda t: bezier(a, b, c, d, t)[1]
    y1 = lambda t: bezier(a1, b1, c1, d1, t)[0]
    Dy = lambda t: bezier(a1, b1, c1, d1, t)[1]

    eq1 = lambda t: (ax - x1(t))**2 + (ay - y1(t))**2
    eq2 = lambda t: (ax - x1(t))**2 + np.power(Dy(t), 2)
    eq3 = lambda t: (ay - y1(t))**2 + np.power(Dx(t), 2)
    eq4 = lambda t: (ax - x1(t)) * Dx(t) + (ay - y1(t)) * Dy(t)

    equations = [eq1, eq2, eq3, eq4]
    min_distance = float('inf')
    best_t = None

    for eq in equations:
        roots = fsolve(eq, (0.0, 1.0))
        for root in roots:
            if 0 <= root <= 1:
                distance = np.sqrt((ax - x1(root))**2 + (ay - y1(root))**2)
                if distance < min_distance:
                    min_distance = distance
                    best_t = root
    C_STATE = {"parameter": best_t, "dist": eq1(best_t)}

    return C_STATE

def discrepancy_test(control_points, points_):
    total_distance = 0.0

    for point in points_:
        closest = closest_point(control_points, point)
        total_distance += closest["dist"]

    return total_distance


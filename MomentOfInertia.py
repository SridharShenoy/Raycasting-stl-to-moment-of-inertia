# assumes uniform mass distribution
import math
import matplotlib.pyplot as plt
import numpy as np
import VTKRayCaster
import plotly.graph_objects as go
import xlsxwriter

'''
Gathering User Information
'''
try:
    height = float(input('Object z-axis height in centimeters? ')) * .01
except ValueError:
    raise ValueError('invalid input')
try:
    mass = float(input('Object mass (g)? ')) * .001
except ValueError:
    raise ValueError('invalid input')
'''
Discretizing stl file using rayCasting to create array of points within solid
'''

RC = VTKRayCaster.rayCaster('sphere copy.stl')
bounds_temp = RC.bounds
bounds = []
minSize = .30


def round_up(x):
    return float(math.ceil(x / minSize)) * minSize


def round_down(x):
    f = float(math.floor(x / minSize)) * minSize
    if f == round_up(x):
        return f - minSize
    else:
        return f


for i in range(6):
    if i % 2 == 1:
        bounds.append(round_up(bounds_temp[i]))
    else:
        bounds.append(round_down(bounds_temp[i]))

p1 = [bounds[0] - 2 * minSize, bounds[2] - 2 * minSize, bounds[4] - 2 * minSize]
p2 = [bounds[1] + 2 * minSize, bounds[3] + 2 * minSize, bounds[5] + 2 * minSize]
p3 = [bounds[1] + 2 * minSize, bounds[2] - 2 * minSize, bounds[5] + 2 * minSize]
print(bounds)


class OctNode:
    def __init__(self, divX, divY, divZ, c1, c2):
        self.Xl = c1[0] - c2[0]
        self.Yl = c1[1] - c2[1]
        self.Zl = c1[2] - c2[2]
        self.Xstep = (self.Xl) / divX
        self.Ystep = (self.Yl) / divY
        self.Zstep = (self.Zl) / divZ
        if self.Xstep < minSize:
            self.Xstep = minSize
        if self.Ystep < minSize:
            self.Ystep = minSize
        if self.Zstep < minSize:
            self.Zstep = minSize
        self.c1 = c1
        self.c2 = c2
        if self.collapse():
            self.leaf = True
            self.children = []
        else:
            self.leaf = False
            ttr = OctNode(divX, divY, divZ, [c1[0], c1[1], c1[2]],
                          [round_up(c1[0] - self.Xl / 2), round_up(c1[1] - self.Yl / 2), round_up(c1[2] - self.Zl / 2)])
            ttl = OctNode(divX, divY, divZ, [round_down(c1[0] - self.Xl / 2), c1[1], c1[2]],
                          [c2[0], round_up(c1[1] - self.Yl / 2), round_up(c1[2] - self.Zl / 2)])
            tbr = OctNode(divX, divY, divZ, [c1[0], round_down(c1[1] - self.Yl / 2), c1[2]],
                          [round_up(c1[0] - self.Xl / 2), c2[1], round_up(c1[2] - self.Zl / 2)])
            tbl = OctNode(divX, divY, divZ, [round_down(c1[0] - self.Xl / 2), round_down(c1[1] - self.Yl / 2), c1[2]],
                          [c2[0], c2[1], round_up(c1[2] - self.Zl / 2)])
            btr = OctNode(divX, divY, divZ, [c1[0], c1[1], round_down(c1[2] - self.Zl / 2)],
                          [round_up(c1[0] - self.Xl / 2), round_up(c1[1] - self.Yl / 2), c2[2]])
            btl = OctNode(divX, divY, divZ, [round_down(c1[0] - self.Xl / 2), c1[1], round_down(c1[2] - self.Zl / 2)],
                          [c2[0], round_up(c1[1] - self.Yl / 2), c2[2]])
            bbr = OctNode(divX, divY, divZ, [c1[0], round_down(c1[1] - self.Yl / 2), round_down(c1[2] - self.Zl / 2)],
                          [round_up(c1[0] - self.Xl / 2), c2[1], c2[2]])
            bbl = OctNode(divX, divY, divZ, [round_down(c1[0] - self.Xl / 2), round_down(c1[1] - self.Yl / 2),
                                             round_down(c1[2] - self.Zl / 2)],
                          [c2[0], c2[1], c2[2]])
            self.children = [ttr, ttl, tbr, tbl, btr, btl, bbr, bbl]

    def collapse(self):
        if self.Xl <= minSize or self.Yl <= minSize or self.Zl <= minSize:
            return True
        x_r = np.arange(self.c2[0] - minSize, self.c1[0] + self.Xstep, self.Xstep)
        y_r = np.arange(self.c2[1] - minSize, self.c1[1] + self.Ystep, self.Ystep)
        z_r = np.arange(self.c2[2] - minSize, self.c1[2] + self.Zstep, self.Zstep)
        for i in x_r:
            for j in y_r:
                if RC.inside([i, j, self.c1[2]], [i, j, self.c2[2]]) > 0:
                    return False
                for k in z_r:
                    if RC.inside([i, self.c1[1], k], [i, self.c2[1], k]) > 0:
                        return False
        return True

    def get_children(self):
        return self.children

    def is_leaf(self):
        return self.leaf

    def is_in(self):
        return self.refp(p1) and self.refp(p2) and self.refp(p3)

    def refp(self, p):
        cmid = [self.c1[0] - self.Xl / 2, self.c1[1] - self.Yl / 2, self.c1[2] - self.Zl / 2]
        return ((RC.inside(self.c1, p) % 2) == 1 and (RC.inside(self.c2, p) % 2) == 1 and (RC.inside(cmid, p2) % 2) == 1
                and (RC.inside([self.c1[0], self.c2[1], self.c2[2]], p2) % 2) == 1
                and (RC.inside([self.c1[0], self.c1[1], self.c2[2]], p2) % 2) == 1
                and (RC.inside([self.c2[0], self.c1[1], self.c2[2]], p2) % 2) == 1
                and (RC.inside([self.c2[0], self.c2[1], self.c1[2]], p2) % 2) == 1
                and (RC.inside([self.c1[0], self.c2[1], self.c1[2]], p2) % 2) == 1
                and (RC.inside([self.c2[0], self.c1[1], self.c1[2]], p2) % 2) == 1)
    def points_in(self):
        c = [[], [], []]
        if self.is_leaf():
            if self.is_in():
                if self.c2[0] == self.c1[0]:
                    x_r = [self.c1[0]]
                else:
                    x_r = np.arange(self.c2[0], self.c1[0] + minSize, minSize)
                if self.c2[1] == self.c1[1]:
                    y_r = [self.c1[1]]
                else:
                    y_r = np.arange(self.c2[1], self.c1[1] + minSize, minSize)
                if self.c2[2] == self.c1[2]:
                    z_r = [self.c1[2]]
                else:
                    z_r = np.arange(self.c2[2], self.c1[2] + minSize, minSize)
                for i in x_r:
                    for j in y_r:
                        for k in z_r:
                            c[0].append(i)
                            c[1].append(j)
                            c[2].append(k)
                return c
            else:
                return c
        else:
            for i in self.children:
                ci = i.points_in()
                c[0] += ci[0]
                c[1] += ci[1]
                c[2] += ci[2]
            return c


o = OctNode(5, 5, 5, [bounds[1], bounds[3], bounds[5]], [bounds[0], bounds[2], bounds[4]])
coords = o.points_in()
print(np.max(coords[0]))
print(np.min(coords[0]))

'''
Brute Force Method (very long O(n^3) runtime, however, most accurate)

coords = [[],[],[]]
x_range = np.arange(bounds[0] - .1, bounds[1] + .1, minSize)
y_range = np.arange(bounds[2] - .1, bounds[3] + .1, minSize)
z_range = np.arange(bounds[4] - .1, bounds[5] + .1, minSize)
for i in x_range:
    for j in y_range:
        for k in z_range:
            if (RC.inside([i, j, k], p1) % 2) == 1 and (RC.inside([i, j, k], p2) % 2) == 1:
                coords[0].append(i)
                coords[1].append(j)
                coords[2].append(k)

if len(coords[0]) == 0:
    raise ValueError('invalid input')
'''
'''
workbook = xlsxwriter.Workbook('test.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(coords):
    worksheet.write_column(row, col, data)

workbook.close()
'''
'''
finding center of mass and rescaling based on user input
'''
x, y, z = coords[0], coords[1], coords[2]
x_mean, y_mean, z_mean = np.mean(x, dtype=np.float64), np.mean(y, dtype=np.float64), np.mean(z, dtype=np.float64)
z_height = np.max(z) - np.min(z)
scale_factor = height / z_height

'''
shifting origin to center of mass and reporting center of mass
'''
x0, y0, z0 = x_mean, y_mean, z_mean
coords[0] = (coords[0] - x0) * scale_factor
coords[1] = (coords[1] - y0) * scale_factor
coords[2] = (coords[2] - z0) * scale_factor
x, y, z = coords[0], coords[1], coords[2]
print('Center of mass coordinates(' + str(-1 * np.min(x)) + ', ' + str(-1 * np.min(y)) +
      ', ' + str(-1 * np.min(z)) + '),')
print('Center of mass coordinates(' + str(x0 * scale_factor) + ', ' + str(y0 * scale_factor) +
      ', ' + str(z0 * scale_factor) + '),')
'''
calculating moment of inertia based on uniform mass density
'''
N = len(coords[1])
print('number of discrete points:' + str(N))
rho = mass / N  # uniform mass density
Ix = sum((coords[1] ** 2 + coords[2] ** 2) * rho)
Iy = sum((coords[0] ** 2 + coords[2] ** 2) * rho)
Iz = sum((coords[0] ** 2 + coords[1] ** 2) * rho)
Ixy = sum((coords[0] * coords[1]) * rho)
Iyz = sum((coords[1] * coords[2]) * rho)
Ixz = sum((coords[0] * coords[2]) * rho)
I = np.array([[Ix, Ixy, Ixz], [Ixy, Iy, Iyz], [Ixz, Iyz, Iz]])
print(I)
'''
scatter plot of points in shifted frame of reference
'''
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
plt.show()
'''
fig = go.Figure(data=[go.Scatter3d(
    x=coords[0],
    y=coords[1],
    z=coords[2],
    mode='markers'
)])

fig.show()

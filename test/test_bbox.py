import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def x_s1(rect):
    return 0.5 * rect.s1 * np.cos(np.radians(rect.angle))


def x_s2(rect):
    return 0.5 * rect.s2 * np.cos(np.radians(90 + rect.angle))


def y_s1(rect):
    return 0.5 * rect.s1 * np.sin(np.radians(rect.angle))


def y_s2(rect):
    return 0.5 * rect.s2 * np.sin(np.radians(90 + rect.angle))


class BoundingBox():

    def __init__(self, center, s1, s2, angle):
        self.x = center[0]
        self.y = center[1]
        self.angle = angle
        self.s1 = s1
        self.s2 = s2
        self.center = Point((
            self.x + x_s1(self) + x_s2(self),
            self.y + y_s1(self) + y_s2(self)))

    def __repr__(self):
        return f'Bounding Box\n' \
            + f'center: {self.x}, {self.y}\n' \
            + f'dimensions: {self.s1} * {self.s2}\n' \
            + f'angle: {self.angle}'

    def plot(self, axis):
        rect = Rectangle(
            (self.x, self.y), self.s1, self.s2, self.angle,
            zorder=10, fill=False)
        axis.add_patch(rect)
        self.center.plot(ax)


class Point():

    def __init__(self, coordinates, color='b'):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.color = color

    def __repr__(self):
        return f'Point\n' \
            + f'coordinates: {self.x}, {self.y}'

    def __add__(self, other):
        return Point((self.x + other.x, self.y + other.y))

    def __subtract__(self, other):
        return Point((self.x - other.x, self.y - other.y))

    def plot(self, axis):
        axis.scatter(self.x, self.y, s=20, c=self.color, zorder=5)


def is_between_lines(line1, line2, point):
    """Return true if point is between parallel lines.

    Args:
        line1: function that returns y for any x
        line2: function that returns y for any x
        point: Point object to check.

    Returns:
        bool if point between parallel lines.

    """
    if not abs(round(line1(-2) - line1(2), 3)) \
            == abs(round(line2(-2) - line2(2), 3)):
        raise ValueError('lines are not parallel!')

    y0, y1, y2 = point.y, line1(point.x), line2(point.x)
    # If line 1 is above line 2
    if y1 > y2:
        return True if y2 < y0 < y1 else False
    else:
        return True if y1 < y0 < y2 else False


def is_in_rect(rect, point):
    """Return true if point is between parallel lines.

    Args:
        line1: function that returns y for any x
        line2: function that returns y for any x
        point: Point object to check.

    Returns:
        bool if point between parallel lines.

    """
    x_s1 = 0.5 * rect.s1 * np.cos(np.radians(rect.angle))
    x_s2 = 0.5 * rect.s2 * np.cos(np.radians(90 + rect.angle))
    y_s1 = 0.5 * rect.s1 * np.sin(np.radians(rect.angle))
    y_s2 = 0.5 * rect.s2 * np.sin(np.radians(90 + rect.angle))

    pt = (rect.center.x + x_s1 + x_s2, rect.center.y + y_s1 + y_s2)
    p1 = Point(pt, color='k')

    pt = (rect.center.x - x_s1 - x_s2, rect.center.y - y_s1 - y_s2)
    p2 = Point(pt, color='b')

    pt = (rect.center.x + x_s1 - x_s2, rect.center.y + y_s1 - y_s2)
    p3 = Point(pt, color='r')

    pt = (rect.center.x - x_s1 + x_s2, rect.center.y - y_s1 + y_s2)
    p4 = Point(pt, color='g')

    def s2_1(x):
        return ((p4.y - p2.y) / (p4.x - p2.x)) * (x - p4.x) + p4.y
    # ax.plot([-2, 2], [s2_1(-2), s2_1(2)])

    def s2_2(x):
        return ((p3.y - p1.y) / (p3.x - p1.x)) * (x - p3.x) + p3.y
    # ax.plot([-2, 2], [s2_2(-2), s2_2(2)])

    def s1_1(x):
        return ((p2.y - p3.y) / (p2.x - p3.x)) * (x - p2.x) + p2.y
    # ax.plot([-2, 2], [s1_1(-2), s1_1(2)])

    def s1_2(x):
        return ((p1.y - p4.y) / (p1.x - p4.x)) * (x - p1.x) + p1.y
    # ax.plot([-2, 2], [s1_2(-2), s1_2(2)])

    return True if is_between_lines(s2_1, s2_2, point)\
        and is_between_lines(s1_1, s1_2, point) else False


if __name__ == "__main__":

    p0 = Point((0, 0), color='k')
    b1 = BoundingBox((p0.x, p0.y), s1=3, s2=5, angle=300)

    __, ax = plt.subplots(1)
    b1.plot(ax)
    plt.axis('equal')

    for xx in range(-7, 7):
        for yy in range(-7, 7):
            point = Point((xx, yy))
            point.color = 'g' if is_in_rect(b1, point) else 'r'
            point.plot(ax)

    plt.show()

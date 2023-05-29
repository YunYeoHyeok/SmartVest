import numpy as np

camera_size = (320, 240)


def distance(bbox1, camera_size):
    x1, y1, x2, y2 = bbox1
    camera_x, camera_y = camera_size
    x_center1, y_center1 = (x1 + x2) / 2, (y1 + y2) / 2
    return np.sqrt((x_center1 - camera_x) ** 2 + (y_center1 - camera_y) ** 2)

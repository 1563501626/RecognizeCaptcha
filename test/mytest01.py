import numpy as np


#  y = wx + b

def compute_error(w, b, points):
    """
    计算损失 ∑(wx+b-y)²
    :param points: np.array
    [0,0,0,...]
    :param w:
    :param b:
    """
    loss = 0
    n = len(points)
    for point in points:
        loss += (1 / n) * (w * point + b - point) ** 2
    return loss

def gradient_descent():
    """
    梯度下降 w = w0 - lr*▽f(θ)
    :return:
    """
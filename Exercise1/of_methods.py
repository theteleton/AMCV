import numpy as np
import cv2
from ex1_utils import gausssmooth, gaussderiv


def lucas_kanade(im1, im2, N):
    kernel = np.ones((N, N))
    I1 = im1
    Ix1, Iy1 = gaussderiv(I1, N // 3)

    I2 = im2
    Ix2, Iy2 = gaussderiv(I2, N // 3)

 
    Ix = (Ix1 + Ix2) / 2 
    Iy = (Iy1 + Iy2) / 2 
    It = I1 - I2

    It = gausssmooth(It, N // 3)
    D = (cv2.filter2D(src=(Ix ** 2), ddepth=-1, kernel=kernel) * cv2.filter2D(src=(Iy ** 2), ddepth=-1, kernel=kernel)) - (cv2.filter2D(src=(Ix * Iy), ddepth=-1, kernel=kernel) ** 2)
 
    U = -((cv2.filter2D(src=(Iy ** 2), ddepth=-1, kernel=kernel) * cv2.filter2D(src=(Ix * It), ddepth=-1, kernel=kernel)) - (cv2.filter2D(src=(Ix * Iy), ddepth=-1, kernel=kernel) * cv2.filter2D(src=(Iy * It), ddepth=-1, kernel=kernel))) / D
    V = -((cv2.filter2D(src=(Ix ** 2), ddepth=-1, kernel=kernel) * cv2.filter2D(src=(Iy * It), ddepth=-1, kernel=kernel)) - (cv2.filter2D(src=(Ix * Iy), ddepth=-1, kernel=kernel) * cv2.filter2D(src=(Ix * It), ddepth=-1, kernel=kernel)))/ D

    return U, V
    

def horn_schunck(im1, im2, n_iters, lmbd):
    # kernel = np.ones((N, N))
    Ld = np.array([[0, 1/4, 0],
                    [1/4, 0, 1/4],
                    [0, 1/4, 0]])
    I1 = im1
    Ix1, Iy1 = gaussderiv(I1, 1)

    I2 = im2
    Ix2, Iy2 = gaussderiv(I2, 1)

    U = np.zeros(im1.shape)
    V = np.zeros(im2.shape)

    Ix = (Ix1 + Ix2) / 2 
    Iy = (Iy1 + Iy2) / 2 
    It = I1 - I2
    It = gausssmooth(It, 1)
    i = 0
    while i <= n_iters:
        Ua = cv2.filter2D(src=U, ddepth=-1, kernel=Ld)
        Va = cv2.filter2D(src=V, ddepth=-1, kernel=Ld)

        P = (Ix * Ua) + (Iy * Va) + It
        D = lmbd + (Ix ** 2) + (Iy ** 2)
 
        U = Ua - (Ix * P / D)
        V = Va - (Iy * P / D)
        i += 1
    
    return U, V


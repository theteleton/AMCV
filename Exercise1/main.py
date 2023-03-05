import numpy as np
import cv2
import matplotlib. pyplot as plt
from ex1_utils import rotate_image, show_flow
from of_methods import lucas_kanade, horn_schunck, lucas_kanade_pyramid, horn_schunck_fast
"""
im1 = cv2.imread("./collision/00000001.jpg")
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2 = cv2.imread("./collision/00000080.jpg")
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

im1 = cv2.imread("./disparity/cporta_left.png")
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2 = cv2.imread("./disparity/cporta_right.png")
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
"""
im1 = cv2.imread("./lab2/001.jpg")
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2 = cv2.imread("./lab2/010.jpg")
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)


# im1 = np.random.rand(200, 200).astype(np.float32)
# im2 = im1.copy()
# im2 = rotate_image(im2, -1)
                   
U_lk, V_lk = lucas_kanade(im1, im2, 7)
#U_hs, V_hs = lucas_kanade(im1, im2, 7)
U_hs, V_hs = horn_schunck(im1, im2, 1000, 0.5)
fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)

ax1_11.imshow(im1)
ax1_12.imshow(im2)

show_flow(U_lk, V_lk, ax1_21, type='angle')
show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)

fig1.suptitle('LK lab2 N=7')
fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
ax2_11.imshow(im1)
ax2_12.imshow(im2)

show_flow(U_hs, V_hs, ax2_21, type='angle')
show_flow(U_hs, V_hs, ax2_22, type='field', set_aspect=True)

fig2.suptitle('LK pyramid lab2 N=3')
plt.show()
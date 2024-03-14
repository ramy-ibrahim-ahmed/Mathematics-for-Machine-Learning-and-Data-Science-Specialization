import numpy as np
import matplotlib.pyplot as plt
import cv2


### READ IMAGE ###
img = cv2.imread("images/leaf_original.png", 0)
print(img)
plt.imshow(img)
plt.title("Image Original", fontsize=18)
plt.show()


### ROTATE THE IMAGE ###
img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
plt.imshow(img_rotated)
plt.title("Image Rotated", fontsize=18)
plt.show()


### SHEAR THE IMAGE ###
rows, cols = img_rotated.shape
M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
img_sheared = cv2.warpPerspective(img, M, (int(cols), int(rows)))
plt.imshow(img_sheared)
plt.title("Image Sheared", fontsize=18)
plt.show()


### ROTATED THEN SEARED ###
img_rotated_sheared = cv2.warpPerspective(img_rotated, M, (int(cols), int(rows)))
plt.imshow(img_rotated_sheared)
plt.title("Image Rotated then Sheared", fontsize=18)
plt.show()


### ROTATED THEN SEARED ###
img_sheared_rotated = cv2.rotate(img_sheared, cv2.ROTATE_90_CLOCKWISE)
plt.imshow(img_sheared_rotated)
plt.title("Image Sheared then Rotated", fontsize=18)
plt.show()
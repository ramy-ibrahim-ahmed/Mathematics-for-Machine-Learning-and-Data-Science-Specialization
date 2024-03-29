import numpy as np
import matplotlib.pyplot as plt
import w3_unittest


### IMAGE DATA ###
img = np.loadtxt(r"data/image.txt")
print("Shape: ", img.shape)
plt.scatter(img[0], img[1], s=0.001, c="black")
plt.title("Oringinal Image", fontsize=18)
plt.show()


### HORIZONTAL SCALING (DILTATION) ###
### FACTOR 2 ###
def T_deltation_2(img):
    M = np.array([[2, 0], [0, 1]])
    return M @ img


img_T_d = T_deltation_2(img)
plt.scatter(img[0], img[1], s=0.001, c="black")
plt.scatter(img_T_d[0], img_T_d[1], s=0.001, c="gray")
plt.title("Horizontal scaling factor 2", fontsize=18)
plt.show()


### REFLECTION ABOUT Y-AXIS (THE VERTICAL AXIS) ###
def T_reflection_yaxis(img):
    M = np.array([[-1, 0], [0, 1]])
    return M @ img


img_T_r = T_reflection_yaxis(img)
plt.scatter(img[0], img[1], s=0.001, c="black")
plt.scatter(img_T_r[0], img_T_r[1], s=0.001, c="grey")
plt.title("Reflection about the vertical axis", fontsize=18)
plt.show()


### STRETCHING BY A SCALER ###
def T_stretch(a, img):
    v = np.array([[a, 0], [0, a]])
    return v @ img


img_stretch = T_stretch(2, img)
plt.scatter(img[0], img[1], s=0.001, c="black")
plt.scatter(img_stretch[0], img_stretch[1], s=0.001, c="grey")
plt.title("Stretch by a scaler", fontsize=18)
plt.show()


### HORIZONATL SHEAR TRANSFORMATION ###
def T_h_shear(s, img):
    M = np.array([[1, s], [0, 1]])
    return M @ img


img_hshear = T_h_shear(2, img)
plt.scatter(img[0], img[1], s=0.001, c="black")
plt.scatter(img_hshear[0], img_hshear[1], s=0.001, c="grey")
plt.title("Horizontal shear transformation", fontsize=18)
plt.show()


### ROTATION ###
def T_rotation(theta, img):
    sin_ = np.sin(theta)
    cos_ = np.cos(theta)
    M = np.array([[cos_, -sin_], [sin_, cos_]])
    return M @ img


theta = np.pi
img_rotated = T_rotation(theta, img)
plt.scatter(img[0], img[1], s=0.001, c="black")
plt.scatter(img_rotated[0], img_rotated[1], s=0.001, c="grey")
plt.title(f"Rotation by {theta * 180 / np.pi:.0f} degree", fontsize=18)
plt.show()


### ROTATION THEN STRETCH ###
def T_rotation_stretch(theta, s, img):
    sin_ = np.sin(theta)
    cos_ = np.cos(theta)
    M_rotated = np.array([[cos_, -sin_], [sin_, cos_]])
    M_stretched = np.array([[s, 0], [0, s]])
    return M_stretched @ (M_rotated @ img)


theta = np.pi
s = 2
img_rotated_stretched = T_rotation_stretch(theta, s, img)
plt.scatter(img[0], img[1], s=0.001, c="black")
plt.scatter(img_rotated_stretched[0], img_rotated_stretched[1], s=0.001, c="grey")
plt.title(f"Rotation by {theta * 180 / np.pi:.0f} degree\nStretch by scaler = {2}",fontsize=18,)
plt.show()
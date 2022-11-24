# from scipy.ndimage import convolve
# https://stackoverflow.com/questions/48097941/strided-convolution-of-2d-in-numpy
import numpy as np
from skimage.util.shape import view_as_windows

def strided4D_v2(arr,arr2,s):
    return view_as_windows(arr, arr2.shape, step=s)
def stride_conv_strided(arr,arr2,s):
    arr4D = strided4D_v2(arr,arr2,s=s)
    return np.tensordot(arr4D, arr2)
filter0 = np.array([
    [
        [-1, -1, -1],
        [1, -1, 1],
        [1, -1, -1],
    ],
    [
        [0, -1, -1],
        [1, 0, -1],
        [1, 0, 0],
    ],
    [
        [0, 0, -1],
        [0, 0, 0],
        [1, 1, 1],
    ]
])
b0 = np.array([2])

filter1 = np.array([
    [
        [-1, -1, 1],
        [0, 1, -1],
        [1, -1, -1],
    ],
    [
        [1, -1, 1],
        [1, 0, -1],
        [-1, -1, 1],
    ],
    [
        [0, -1, 0],
        [1, 1, 1],
        [-1, 0, 1],
    ]
])
b1 = np.array([3])

# filter_list = [filter0, filter1]
# b_list = [b0, b1]

# should be ideally 4 shape
# 1st should be batch size
input_tensor = np.array([
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2, -1, 2, 1, 2, 0],
        [0, 0, -1, 2, 0, -1, 0],
        [0, -1, 1, -1, 0, 1, 0],
        [0, 2, 2, 0, 2, 1, 0],
        [0, 0, 2, 2, 2, -1, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 1, 2, 0, 0],
        [0, 2, -1, 0, 0, 0, 0],
        [0, -1, 0, 1, -1, 0, 0],
        [0, 1, 2, -1, 2, 0, 0],
        [0, 2, 2, -1, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 1, 2, 0],
        [0, -1, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, -1, 1, 2, 2, -1, 0],
        [0, -1, -1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
])
print(input_tensor.shape)
stride = 2
output0 = np.zeros((3,3))
output1 = np.zeros((3,3))
for it in range(input_tensor.shape[0]):
    # print(input_tensor[it, :,:])
    inter = stride_conv_strided(input_tensor[it, :,:], filter0[it, :, :], stride) 
    output0 += inter 

output0 += b0
print(f"output by filter0 = {output0}")

for it in range(input_tensor.shape[0]):
    inter = stride_conv_strided(input_tensor[it, :,:], filter1[it, :, :], stride)
    output1 += inter 
output1 += b1
print(f"output by filter1 = {output1}")

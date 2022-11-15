import numpy as np
import math
import yuvio

yuv_frame = yuvio.mimread("C:/Users/Dimitris/Desktop/Beauty_1920x1080_120fps_420_8bit_YUV.yuv", 1920, 1080, "yuv420p")

#print(yuv_frame[0].y)


#https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))



import math
def DCT(A,X):
    Y = np.dot(A,X)
    A = np.transpose(A)
    print(np.dot(Y,A))
    
 

# def dctTransform(matrix):
# m = 4
# n = 4
#     # dct will store the discrete cosine transform
#     dct = []
#     for i in range(m):
#         dct.append([None for _ in range(n)])
 
#     for i in range(m):
#         for j in range(n):
 
#             # ci and cj depends on frequency as well as
#             # number of row and columns of specified matrix
#             if (i == 0):
#                 ci = 1 / (m ** 0.5)
#             else:
#                 ci = (2 / m) ** 0.5
#             if (j == 0):
#                 cj = 1 / (n ** 0.5)
#             else:
#                 cj = (2 / n) ** 0.5
 
#             # sum will temporarily store the sum of
#             # cosine signals
#             sum = 0
#             for k in range(m):
#                 for l in range(n):
 
#                     dct1 = matrix[k][l] * math.cos((2 * k + 1) * i * pi / (
#                         2 * m)) * math.cos((2 * l + 1) * j * pi / (2 * n))
#                     sum = sum + dct1
 
#             dct[i][j] = ci * cj * sum
 
#     for i in range(m):
#         for j in range(n):
#             print(dct[i][j], end="\t")
#         print()




matrix = [[5,11,8,10],[9,8,4,12],[1,10,11,4],[19,6,15,7]]

a = 0.5
b = (0.5**0.5)*math.cos(math.pi/8)
c = (0.5**0.5)*math.cos(3*math.pi/8)
dct_arr = [[a,a,a,a],[b,c,c*(-1),b*(-1)],[a,a*(-1),a*(-1),a],[c,b*(-1),b,c*(-1)]]

#DCT(dct_arr,matrix)
 
c = yuv_frame[0].y
c = blockshaped(c, 4, 4)



for frame in c:
    DCT(dct_arr,frame)
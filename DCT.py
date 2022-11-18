import numpy as np
import math
import yuvio

SIZE = 4

yuv_frame = yuvio.mimread("C:/Users/Dimitris/Desktop/Beauty_1920x1080_120fps_420_8bit_YUV.yuv", 1920, 1080, "yuv420p")

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

def transform(A,X):
    Y = np.dot(A,X)
    A = np.transpose(A)
    return np.dot(Y,A)

#fixed DCT HVEC array
a_0 = 64
b_0 = 83
b_1 = 36
dct_arr = [[a_0,a_0,a_0,a_0],
           [b_0,b_1,b_1*(-1),b_0*(-1)],
           [a_0,a_0*(-1),a_0*(-1),a_0],
           [b_1,b_0*(-1),b_0,b_1*(-1)]]

#fixed DST HVEC array
dst_arr = [[29,55,74,84],
           [74,74,0,-74],
           [84,-29,-74,55],
           [55,-84,74,-29]]

#take first frame of y
first_y_frame = yuv_frame[0].y
    
#reform it into 4x4 lists
first_y_frame = blockshaped(first_y_frame, SIZE, SIZE)

#dct transform on the matrix
transformed_matrix = []
for frame in first_y_frame:
    transformed_matrix.append(transform(dct_arr,frame))

#calculate energy
avg_energy = [[0] * SIZE for _ in range(SIZE)]
avg_energy = np.array(avg_energy, dtype=np.int64)

for frame in transformed_matrix:
    for i in range(len(frame)):
        for j in range(len(frame[0])):
            avg_energy[i][j] += (frame[i][j]//128)**2

for i in range(len(frame)):
    for j in range(len(frame[0])):
        avg_energy[i][j] = avg_energy[i][j]//len(first_y_frame)

#reform array to 4x4
avg_energy = np.array(avg_energy).reshape((SIZE, -1))
     
#calculate energy   
initial_avg_energy = [[0] * SIZE for _ in range(SIZE)]
initial_avg_energy = np.array(initial_avg_energy, dtype=np.int64)

for frame in first_y_frame:
    for i in range(len(frame)):
        for j in range(len(frame[0])):
            initial_avg_energy[i][j] += frame[i][j]**2

for i in range(len(frame)):
    for j in range(len(frame[0])):
        initial_avg_energy[i][j] = initial_avg_energy[i][j]//len(first_y_frame)

#reform array to 4x4
initial_avg_energy = np.array(initial_avg_energy).reshape((SIZE, -1))

#dst transform on the patrix
transformed_matrix_dst = []
for frame in first_y_frame:
    transformed_matrix_dst.append(transform(dst_arr,frame))

#calculate energy
avg_energy_dst = [[0] * SIZE for _ in range(SIZE)]
avg_energy_dst = np.array(avg_energy_dst, dtype=np.int64)

for frame in transformed_matrix_dst:
    for i in range(len(frame)):
        for j in range(len(frame[0])):
            avg_energy_dst[i][j] += (frame[i][j]//128)**2

#calculate avergae energy
for i in range(len(frame)):
    for j in range(len(frame[0])):
        avg_energy_dst[i][j] = avg_energy_dst[i][j]//len(first_y_frame)

#reform array to 4x4
avg_energy_dst = np.array(avg_energy_dst).reshape((SIZE, -1))
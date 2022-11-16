import numpy as np
import math
import yuvio

SIZE = 4

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



def DCT(A,X):
    Y = np.dot(A,X)
    A = np.transpose(A)
    #print(np.dot(Y,A))
    return np.dot(Y,A)

#generates the array for operations for each transform
def generate_DST_or_DCT(flag,n):
    
    arr = []
    if flag :
        
        for i in range(n):
            for j in range(n):
                if i == 0:
                  P = 1
                else:
                  P = 2**0.5
              
                arr.append(P/(n**0.5)*math.cos((math.pi)/n*(j + 0.5)*i))
    else:
        for i in range(n):
            for j in range(n):
                arr.append(round(128*(2/((2*n + 1)**0.5))*math.sin((2*i + 1)*(j + 1)*math.pi)/(2*n + 1)))

    
    return np.array(arr).reshape((SIZE, -1))

dct_arr = generate_DST_or_DCT(1,SIZE)

#take first frame of y
c = yuv_frame[0].y
#reform it into 4x4 lists
c = blockshaped(c, 4, 4)

#dct transform on the patrix
transformed_matrix = []

for frame in c:
    transformed_matrix.append(DCT(dct_arr,frame))

#calculate energy
energy = [[0] * 4 for _ in range(4)]
for frame in transformed_matrix:
    for i in range(len(frame)):
        for j in range(len(frame[0])):
            energy[i][j] += (abs(frame[i][j])**2)

#calculate avergae energy
avg_energy = []
for val in energy:
    for v in val:
        avg_energy.append(v//len(c))

#reform array to 4x4
avg_energy = np.array(avg_energy).reshape((4, -1))
     
#calculate energy   
initial_energy = [[0] * 4 for _ in range(4)]
for frame in c:
    for i in range(len(frame)):
        for j in range(len(frame[0])):
            initial_energy[i][j] += (abs(frame[i][j])**2)

#calculate avergae energy
initial_avg_energy = []
for val in initial_energy:
    for v in val:
        initial_avg_energy.append(v//len(c))

#reform array to 4x4
initial_avg_energy = np.array(initial_avg_energy).reshape((4, -1))
import numpy as np
import yuvio

#size of transform tables
SIZE = 4

yuv_frame = yuvio.mimread("C:/Users/Dimitris/Desktop/Beauty_1920x1080_120fps_420_8bit_YUV.yuv", 1920, 1080, "yuv420p")

#https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
#@PARAMS --> The array to be sized down, the number of rows and cols of the arrays to be created
#@RETURN --> The initial array seperated on subarrays of the desired dimensions
#@DESC   --> Transform an array into subarrays of a given size
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

#@PARAMS --> Two matrixes A,X where X is the array to be tranformed and A is the transformation array based on HVEC standards
#@RETURN --> The transformed array on numpy array form
#@DESC   --> The transform procedure, performing serial matrix multiplication
def transform(A,X):
    
    #transform procedure based on HVEC standards
    Y = np.dot(A,X)
    A = np.transpose(A)
    return np.dot(Y,A)

#@PARAMS --> Transformed matrix and frame length
#@RETURN --> Array of matrix average on numpy form
#@DESC   --> Average energy calculation of transformed matrix
def calculate_transform_energy_average(transformed_matrix,frame_length):
    
    #normilization value based on HVEC standards
    NORMALIZATION_VALUE = 2**14
    
    #2d array intialized to zero for the calculation of energy
    avg_energy = [[0] * SIZE for _ in range(SIZE)]
    #change array size to 64 bit integer to avoid overflows
    avg_energy = np.array(avg_energy, dtype=np.int64)

    #energy calculation
    for frame in transformed_matrix:
        for i in range(len(frame)):
            for j in range(len(frame[0])):
                #divizion with 2^14 for data normalization
                avg_energy[i][j] += (frame[i][j]//NORMALIZATION_VALUE)**2

    #mean of each cell calculation
    for i in range(len(frame)):
        for j in range(len(frame[0])):
            avg_energy[i][j] = avg_energy[i][j]//frame_length

    #reform array to 4x4
    return np.array(avg_energy).reshape((SIZE, -1))

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

#DCT transform on each of the 4x4 arrays of the first_y_frame
transformed_dct_matrix = []
for frame in first_y_frame:
    transformed_dct_matrix.append(transform(dct_arr,frame))
    
#DST transform on each of the 4x4 arrays of the first_y_frame
transformed_dst_matrix = []
for frame in first_y_frame:
    transformed_dst_matrix.append(transform(dst_arr,frame))

#dst and dct average energy calculation
avg_dst_energy = calculate_transform_energy_average(transformed_dst_matrix, len(first_y_frame))
avg_dct_energy = calculate_transform_energy_average(transformed_dct_matrix,len(first_y_frame))
     
#2d array intialized to zero for the calculation of energy
initial_avg_energy = [[0] * SIZE for _ in range(SIZE)]
#change array size to 64 bit integer to avoid overflows
initial_avg_energy = np.array(initial_avg_energy, dtype=np.int64)

#energy calculation of the intial frame
for frame in first_y_frame:
    for i in range(len(frame)):
        for j in range(len(frame[0])):
            initial_avg_energy[i][j] += frame[i][j]**2

#mean of each cell calculation
for i in range(len(frame)):
    for j in range(len(frame[0])):
        initial_avg_energy[i][j] = initial_avg_energy[i][j]//len(first_y_frame)

#reform array to 4x4
initial_avg_energy = np.array(initial_avg_energy).reshape((SIZE, -1))

print(avg_dct_energy,avg_dst_energy,initial_avg_energy)
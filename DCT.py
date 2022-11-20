import numpy as np
import yuvio

#size of transform tables
SIZE = 4

yuv_frame = yuvio.mimread("C:/Users/Dimitris/Desktop/Beauty_1920x1080_120fps_420_8bit_YUV.yuv", 1920, 1080, "yuv420p")

#@PARAMS -->    arr: integer array   (array to be sized down)
#            n_rows: integer        (number of rows)
#            n_cols: integer        (number of cols)
#@RETURN --> 2d array containing 2d arrays  (initial array seperated on subarrays of the desired dimensions)
#@DESC   --> Transform an array into subarrays of a given size
def blockshaped(arr, n_rows, n_cols):
    #store array dimensions
    height, width = arr.shape

    #compatability check
    assert height % n_rows == 0, f"{h} rows is not evenly divisible by {n_rows}"
    assert width % n_cols == 0, f"{w} cols is not evenly divisible by {n_cols}"

    #first reshape  --> number of arrays subarrays, number of internal subarrays of subarrays
    #                   newshape spesial variable (docs: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy.reshape),
    #                   number of rows
    #                   result reshape => array in a form of n_rows x n_cols subarrays
    #swapaxes       --> break the subarrays into the desired blocks  ex. for array:  [[ 0  1  2  3  4  5]      \                     [[ 0  6 12 18]
    #                                                                                 [ 6  7  8  9 10 11]        >  swapaxes(0,1) =>  [ 1  7 13 19]
    #                                                                                 [12 13 14 15 16 17]        >                    [ 2  8 14 20]
    #                                                                                 [18 19 20 21 22 23]]     /                      [ 3  9 15 21]
    #                                                                                                                                 [ 4 10 16 22]
    #                                                                                                                                 [ 5 11 17 23]]
    #                   docs: https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html
    #second reshape --> better format the array subarrays to the desired dimensions
    return (arr.reshape(height//n_rows, n_rows, -1, n_cols)
               .swapaxes(1,2)
               .reshape(-1, n_rows, n_cols))

#@PARAMS --> X: integer array         (array to be transformed)
#            A: integer array         (transformation array based on HVEC standards)
#@RETURN --> integer numpy array      (he transformed array)
#@DESC   --> The matrix transformation procedure, performing matrix multiplication based on HVEC standards
def transform(A,X):

    Y = np.dot(A,X)
    A = np.transpose(A)
    return np.dot(Y,A)

#@PARAMS --> transformed_matrix: integer array  (transfomrmed array)
#            frame_length: integer              (length of processing frame)
#@RETURN --> integer_64 numpy array             (average array)
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
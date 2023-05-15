import numpy as np

# cheat file made with copilot

# create a 6x6 matrix
matrix = np.array([
    8,4,2,3,9,1,
    0,6,4,3,7,10,
    2,0,1,3,5,8,
    6,3,1,4,8,2,
    0,2,4,3,5,9,
    2,7,9,1,3,5]).reshape(6,6)

# create a 3x3 kernel
kernel = np.array([2,0,3,
                   1,1,0,
                   3,-1,4]).reshape(3,3)

print(matrix)
print(kernel)

# calculate the convolution
def convolution(matrix, kernel):
    # get the dimensions of the matrix and the kernel
    m_rows, m_cols = matrix.shape
    k_rows, k_cols = kernel.shape
    
    # calculate the dimensions of the output matrix
    output_rows = m_rows - k_rows + 1
    output_cols = m_cols - k_cols + 1
    
    # create the output matrix
    output = np.zeros((output_rows, output_cols))
    
    # iterate over the matrix
    for row in range(output_rows):
        for col in range(output_cols):
            # get the current matrix
            matrix_slice = matrix[row:row+k_rows, col:col+k_cols]
            
            # perform the dot product
            output[row, col] = np.sum(matrix_slice * kernel)
    
    return output

# calculate the maxpool
def maxpool(matrix, pool_size):
    # get the dimensions of the matrix
    m_rows, m_cols = matrix.shape
    
    # calculate the dimensions of the output matrix
    output_rows = m_rows // pool_size
    output_cols = m_cols // pool_size
    
    # create the output matrix
    output = np.zeros((output_rows, output_cols))
    
    # iterate over the matrix
    for row in range(output_rows):
        for col in range(output_cols):
            # get the current matrix
            matrix_slice = matrix[row*pool_size:row*pool_size+pool_size, col*pool_size:col*pool_size+pool_size]
            
            # perform the maxpool
            output[row, col] = np.max(matrix_slice)
    
    return output

# calculate the convolution
output = convolution(matrix, kernel)
print(output)

# calculate the maxpool
output = maxpool(output, 2)
print(output)

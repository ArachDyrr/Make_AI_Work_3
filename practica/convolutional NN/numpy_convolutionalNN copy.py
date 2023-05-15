import numpy as np

# copy of the cheat file made with copilot, copilot is not used to create the padding. 

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

print(f'The 6*6 matrix: \n{matrix} \n \n The 3*3 kernel: \n{kernel} \n ')

# add padding to the matrix
padding = 2 
matrix_rows, matrix_cols = matrix.shape

# define the add_padding matrix function
def add_padding (matrix, padsize):
    padded_matrix = np.pad(matrix, padsize, mode='constant', constant_values=0)
    return padded_matrix

matrix = add_padding(matrix, padding)

print(f'''The new ({matrix_cols+2*padding},{matrix_rows+2*padding}) matrix: \n{matrix} \n ''')


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
print(f'''The Output after the first convolution
{output}''')

# calculate the maxpool
maxpool_dimentions = 2
output = maxpool(output, maxpool_dimentions)
print(f'''
The output after maxpool ({maxpool_dimentions},{maxpool_dimentions}) has been applied: 
{output}''')

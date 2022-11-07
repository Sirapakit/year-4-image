import numpy as np

import time
start_time = time.time()

box = (1/256) * np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
])

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(box, k = 2)
sigma = np.diag(sigma)
temp = np.dot(U, sigma)
temp_trans = np.transpose(temp[:,1])
new_temp_trans = np.array([temp_trans]).transpose()
new_Vt = np.array([Vt[1,:]])
print(f'The original matirx is {box}')
print(f'The matrix by SVD method is {np.dot(new_temp_trans,new_Vt)}')

print("--- %s seconds ---" % (time.time() - start_time)) 

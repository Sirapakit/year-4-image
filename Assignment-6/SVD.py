import numpy as np
from numpy.linalg import svd

box_filter = np.array([
    [3, 6, 9],
    [4, 8, 12],
    [5, 10, 15]
])

U, S, VT = svd(box_filter, full_matrices=True)
print(f'Left Singular Vectors is {U}')
print(f'Singular Vectors is {S}')
print(f'Right Singular Vectors is {np.transpose(VT)}')

S_diag = np.zeros((3,3))
S_diag[0,0] = S[0]
S_diag[1,1] = S[1]
S_diag[2,2] = S[2]
print(f'smat matrix is {S_diag}')

from numpy.linalg import multi_dot
print('Proof')
print(multi_dot([U,S_diag,VT]),'\n',box_filter)
temp = multi_dot([S_diag,VT])
print('Proof 2')
print(multi_dot([U,temp]),'\n',box_filter)

print(temp.ndim)

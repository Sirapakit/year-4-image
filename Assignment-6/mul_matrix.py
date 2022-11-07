import numpy as np

A = np.array([[0.3268,0.1307,0.19609,0.1307,0.3268]])
B = np.array([[0.11952286,0.47809144,0.71713717,0.47809144,0.11952286]])
A = np.transpose(A)

print(A.ndim)
print(B.ndim)

print(A.shape)
print(B.shape)

print(np.dot(A,B))

temp = [1,2,3,4,5,6,7,8,9]
print(type(temp))
print(max(temp))
# import pandas as pd

# df = pd.read_csv('../SampleCSVFile_2kb.csv', delimiter=',',encoding= 'unicode_escape')

# list_of_csv = [list(row) for row in df.values]

# print(list_of_csv)


# from random import random as rand
# n_channels = 23
# my = [rand() for _ in range(n_channels)]
# print(type(my))
# print(len(my))
# print(my)




# import numpy as np

# input_list =    [9,8,7,
#                 6,5,4,
#                 3,2,1]
# print(type(input_list))

# new_list = []
# weighted_matrix = np.array([[1,2,3],
#                             [4,5,4],
#                             [3,2,1]])
# weighted_list = weighted_matrix.flatten().tolist()
# print(weighted_list)

# for i in range(len(input_list)):
#         new_list.extend([input_list[i] for a in range(weighted_list[i])])

# new_list.sort()
# print(f'The new_list is {new_list}')


from itertools import repeat
list = [47, 47, 47, 48, 48, 48, 255, 255, 255, 255, 47, 47, 47, 47, 47, 255, 255, 255, 255, 0, 0, 0, 46, 46, 46]
print ("Printing the numbers repeatedly : ")   

print(len(list))
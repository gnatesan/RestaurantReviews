

import numpy as np

arr1 = np.array(["cat", "in", "hat"])

arr2 = np.array(["the", "cat", "in", "the", "cat"])

arr = np.concatenate((arr1, arr2))
new_arr, count = np.unique(arr, return_counts=True)
print(new_arr)
print(count)
print(arr)

indexes = []
ans = np.array([])
print(len(new_arr))
for i in range(len(new_arr)):
    if (count[i] > 3):
        print("reached")
        indexes.append(i)
#print(indexes)

#for i in indexes:
#    new_arr = np.delete(new_arr, i)
#print(len(new_arr))


x = np.array([0,10,20,30,40,50,60])
exclude = [1, 3, 5]
a = np.delete(x, exclude, axis=0)


print(a)









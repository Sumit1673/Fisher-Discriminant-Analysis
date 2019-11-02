import pandas as pd
import numpy as np
from fisher_lda import fisher_ld
from copy import deepcopy as dp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
data = pd.read_csv("sample3.csv", delimiter=',')
data_mat = np.zeros([10,9])
colm =0
for each_col in data:
    data_mat[:,colm] = data[each_col]
    colm +=1
data_mat = np.round(data_mat,2)
row = np.size(data_mat,0)
col = np.size(data_mat,1)
print(row, col)
d = col/colm
# Find optimal w for categories omega1 and omega2.
class_numbers = [2, 3]

Y_class, p, X_class = fisher_ld(data_mat, 3, class_numbers)
# Given
w = np.array([1.0, 2.0, -1.5])
X_cls_arr = np.zeros([10,9])
for e_cls in class_numbers:
    b = np.array(X_class[str(e_cls)])
    col = 3*(int(e_cls) - 1)

    for i in range(3):
        for j in range(10):
            vb = b[:, j]
            X_cls_arr[j,col] = dp(vb[0, i])
        col+=1

# Plot a line representing w and the positions of the plotted points.

fig = plt.figure()
plt.title(" Plot for the discriminant vector")
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_cls_arr[:, 3], X_cls_arr[:, 4],X_cls_arr[:, 5], c='r', marker='o', s=400)
ax.scatter(X_cls_arr[:, 6], X_cls_arr[:, 7],X_cls_arr[:, 8], c='b', marker='x', s=300)
plt.savefig('Figure1_Discriminant_Vector.png')
# plt.show()

fig2 = plt.figure()
plt.title(" Projection of Ouput class")
project_vect = np.zeros([10,9])
project_vect[:, 3:5] = np.transpose(Y_class['2'])*w[0]
project_vect[:, 6:8] = np.transpose(Y_class['2'])*w[0]

ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(project_vect[:, 3], project_vect[:, 4], project_vect[:, 5], c='b', marker='o', s=400)
ax2.scatter(project_vect[:, 6], project_vect[:, 7], project_vect[:, 8], c='r', marker='+', s=300)
plt.savefig('Projection_Vector.png')

# Fit each distribution with a univariate Gaussian.
mu_2 = np.mean(Y_class['2'])
sigma_2 = np.std(Y_class['2'])
mu_3 = np.mean(Y_class['3'])
sigma_3 = np.std(Y_class['3'])

# Find decision boundary.
y_0 = 0.06

# Calculate training error.
Y_Data = np.hstack([Y_class['2'], Y_class['3']])
L_Data = np.hstack([2*np.ones([1,row]), 3* np.ones([1,row])])

Decision = Y_Data < y_0
Decision = Decision.astype(np.int)
Decision = Decision + 2

classification_per = 100 *(np.sum(Decision == L_Data)/L_Data.size)
print(classification_per)

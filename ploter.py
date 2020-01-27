import numpy as np
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
colors=[]
for key in mcd.BASE_COLORS.keys():
    colors.append(key)
# print(colors)
clrs=[]
for key in mcd.CSS4_COLORS.keys():
    clrs.append(key)

def plot(matrix):
    for i in range(len(matrix)):
        plt.scatter(matrix[i][0], matrix[i][1], color=colors[int(matrix[i][2])])
def plot_c(matrix):
    for i in range(len(matrix)):
        plt.scatter(matrix[i][0], matrix[i][1], color='b')
def plot_clus(matrix):
    for i in range(len(matrix)):
        plt.scatter(matrix[i][0], matrix[i][1], color=clrs[int(matrix[i][2])])
def show():
    plt.show()
def save(name):
    plt.savefig(name+".png")

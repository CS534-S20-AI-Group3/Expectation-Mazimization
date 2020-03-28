from common_functions import *
from em_with_plot import *
import random
import numpy
import sys
from matplotlib import pyplot as plt
from expect_maximum_clusters import *
from matplotlib import animation


def k_means_clustering(data, num_clust):
    data_set = read_board(data)
    init_clusters = init_k_means(data_set, int(num_clust))
    clusters = k_means(data_set, init_clusters)
    c_string = 'rgbyc'
    plt.figure(1)
    ax1 = plt.subplot()
    i = 0
    if(len(data_set[0])==2):
        for c in clusters:
            for p in c:
                # print(c_string[i])
                ax1.plot(p[0], p[1], color=str(c_string[i]), marker='o', markersize=15, alpha=0.4, mec=str(c_string[i]),
                         mew=0)
            i = i + 1
        ax1.set_title('K-means')


    print(len(clusters))

file_path = r"/home/ankit/git/AI/assignment_2/Expectation-Mazimization/sample_EM_data.csv"
no_of_clusters = 3
k_means_clustering(file_path,no_of_clusters)
em_clustering(file_path,no_of_clusters)
plt.show()

# if __name__ == "__main__":
#     if len(sys.argv) == 3:
#         expect_max(sys.argv[1], sys.argv[2])
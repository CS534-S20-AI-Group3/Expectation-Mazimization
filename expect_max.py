from common_functions import *
from expect_maximum_clusters import *
import random
import numpy
import sys


def expect_max(data, num_clust):
    data_set = read_board(data)
    init_clusters = init_k_means(data_set, int(num_clust))
    clusters = k_means(data_set, init_clusters)
    print(clusters)
    print(len(clusters))

if __name__ == "__main__":
    if len(sys.argv) == 3:
        expect_max(sys.argv[1], sys.argv[2])
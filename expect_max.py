from em_with_plot import *
import random
import numpy
import sys
from matplotlib import pyplot as plt
from expect_maximum_clusters import *
import time

def em_restart(given_points,no_k):

    start_time = time.perf_counter()
    no_of_clusters = int(no_k)
    # means_k = k_means_clustering(file_path,no_of_clusters)
    # print(" Passing means from K means to EM ")
    # for i in range(2,12):
    sol_ll = 0
    sol_clusters = []
    sol_em = []
    lowest_bic = 0
    best_k = 0
    restart = True
    if(no_of_clusters==0):
        bic_bool=True
    else:
        bic_bool = False


    print("no of clusters", no_of_clusters)
    print("Number of Dimensions", len(given_points[0]))
    print("NUmber of points", len(given_points))
    if(no_of_clusters!=0):
        print("Running EM with restart")
        means_k = k_means_clustering(given_points,no_of_clusters)
        sol = em_clustering_kmean(given_points, no_of_clusters, bic_bool, means_k)
        sol_ll = sol[0]
        sol_clusters = sol[1]
        sol_em = sol[2]
        while (restart):
            end_time = time.perf_counter()
            if (end_time - start_time > 9):
                restart = False
                print("best ll ", sol_ll)
                # print("clusters", sol_clusters)
                print("EM mean , covar , weight", sol_em)
                print("end time", end_time - start_time)
                break
            sol = em_clustering(given_points, no_of_clusters, bic_bool)
            #print("ll",sol[0],sol[1],sol[2])
            if(sol[0]>sol_ll or sol_ll==0):
                sol_ll = sol[0]
                sol_clusters = sol[1]
                sol_em = sol[2]



    else:
        print("Running BIC")
        start_k = 1
        run = True
        while(run):
            end_time1 = time.perf_counter()
            if (end_time1 - start_time > 9 or start_k == 20):
                run = False
                print("best k found", best_k)
                print("best ll ", sol_ll)
                #print("clusters", sol_clusters)
                print("EM mean , covar , weight", sol_em)
                print("end time", end_time1 - start_time)
                break
            means_k = k_means_clustering(given_points,start_k)
            sol = em_clustering_kmean(given_points, start_k, bic_bool, means_k)
            #sol = em_clustering(given_points, start_k, bic_bool)
            #print("ll", sol[0], sol[1], sol[2])
            bic = (start_k)*(np.log(len(given_points)))-(2*sol[0])
            print("BIC for k ",start_k,bic)
            if(lowest_bic==0 or bic <lowest_bic):
                best_k = start_k
                sol_ll = sol[0]
                sol_clusters = sol[1]
                sol_em = sol[2]
                lowest_bic = bic
            start_k = start_k+1
        end_time2 = time.perf_counter()
        if(end_time2 - start_time <9.2):
            while (restart):
                end_time3 = time.perf_counter()
                if (end_time3 - start_time > 9.8):
                    restart = False
                    print("best k found",best_k)
                    print("best ll ", sol_ll)
                    print("clusters", sol_clusters)
                    print("EM mean , covar , weight", sol_em)
                    print("end time", end_time3 - start_time)
                    break
                sol = em_clustering(given_points, best_k,False)
                # print("ll",sol[0],sol[1],sol[2])
                if (sol[0] > sol_ll or sol_ll == 0):
                    sol_ll = sol[0]
                    sol_clusters = sol[1]
                    sol_em = sol[2]


    if (len(given_points[0]) != 1):
        plt.figure(2)
        ax = plt.subplot()
        color_bar = []
        for e in range(len(sol_em)):
            f = (random.uniform(0.3, 1), random.uniform(0, 1), random.uniform(0, 1))
            color_bar.append(f)
        c = 0
        for i in sol_clusters:
            for j in i:
                ax.plot(j[0], j[1], color=color_bar[c], marker='o', markersize=10,
                        alpha=0.4,
                        mec=color_bar[c], mew=0)
            c = c+1

        i = 0
        for e in sol_em:
            #     # circle2 = plt.Circle((e[0][0], e[0][1]),radius=e[1],color='green',fill=False)
            #     # ax.add_artist(circle2)
            ax.plot(e[0][0], e[0][1], color=color_bar[i], marker='d', markersize=8, markeredgecolor='k',
                    markeredgewidth=1)
            i = i + 1
        #     # print("final circle plotted")
        ax.set_title('EM , LL ' + str(sol_ll))
        plt.show()
    if (len(given_points[0]) == 1):
        plt.figure(2)
        ax = plt.subplot()
        color_bar = []
        for e in range(len(sol_em)):
            f = (random.uniform(0.3, 1), random.uniform(0, 1), random.uniform(0, 1))
            color_bar.append(f)
        c = 0
        for i in sol_clusters:
            for j in i:
                ax.plot(j[0], j[0], color=color_bar[c], marker='X', markersize=10,
                        alpha=0.4,
                        mec=color_bar[c], mew=0)
            c = c+1

        i = 0
        for e in sol_em:
            #     # circle2 = plt.Circle((e[0][0], e[0][1]),radius=e[1],color='green',fill=False)
            #     # ax.add_artist(circle2)
            ax.plot(e[0][0], e[0][0], color=color_bar[i], marker='d', markersize=8, markeredgecolor='k',
                    markeredgewidth=1)
            i = i + 1
        #     # print("final circle plotted")
        ax.set_title('EM , LL ' + str(sol_ll))
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 3:
        given_points = np.array(read_board(sys.argv[1]))
        em_restart(given_points, sys.argv[2])


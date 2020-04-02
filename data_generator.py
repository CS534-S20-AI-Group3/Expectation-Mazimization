import random
import sys
import csv

def generate_data(num_points):
    with open('close_clusters.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter= ',', quotechar='|', quoting = csv.QUOTE_MINIMAL)
        for x in range(int(int(num_points)/2)):
            filewriter.writerow([random.uniform(10, 11), random.uniform(10, 12)])
        for x in range(int(int(num_points)/2)):
            filewriter.writerow([random.uniform(10, 11), random.uniform(11, 14)])


if __name__ == "__main__":
    generate_data(sys.argv[1])
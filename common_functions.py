import csv
import math
import numpy

# Reads a given CSV file to generate a list of tuples
def read_board(filename):
  init_data = []
  data = []
  # Read the CSV file
  with open(filename, 'r', encoding='utf-8-sig') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        temp_row =[]
        for el in row:
            temp_row.append(float(el))
        init_data.append(temp_row)
    #print(init_data)
  #for row in init_data:
      #data.append((float(row[0]), float(row[1])))
  return init_data

# Calculates the distance between 2 tuples
def distance(tup1, tup2):
    xdiff = abs(tup1[0]-tup2[0])
    ydiff = abs(tup1[1]-tup2[1])
    dist = math.sqrt((xdiff**2)+(ydiff**2))
    return dist

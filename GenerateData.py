#the purpose of this file is to generate the data points needed for TSP
import numpy as np
import random as rand
import math
import argparse



'''
command line use:

python GenerateData.py [coords_type] [num_cities] [out_file_name_prefix]

Inputs:
    num_cities: 
        - int, the number of city coordinates you want to generate (coordinates b/w -100,100)
    out_file_name_prefix: 
        - str, the name of the file you want to output city coordinates and pairwise distances to.
        - pairwise distances will write to out_file_name_prefix_dist.csv
        - coordinates will write to out_file_name_prefix.coords
     

''' 

def GenerateCoords(filename,num):
    #generates a bunch of random coordinates between -100 and 100 for N = num cities, saves in filename
    coords = open(filename, "w")

    for i in range(num):
        x = rand.uniform(-100,100)
        y = rand.uniform(-100,100)
        coords.write(str(x) + ","+ str(y) +"\n")
    coords.close()



def GenerateCircleCoords(filename,num):
    coords = open(filename, "w")

    for i in range(num):
        angle = (2 * math.pi * i) / num
        x = 100 * math.cos(angle)
        y = 100 * math.sin(angle)
        coords.write(str(x) + ","+ str(y) +"\n")
    return
#Calculates distances between all cities and writes them to a file in a distance matrix format
def CalcDistances(input_file, output_file):

    #get data
    coords = np.genfromtxt(input_file, delimiter=',')
    #write to file
    dist = open(output_file, "w")

    N = coords.shape[0]
    #create initial matrix
    D = np.zeros((N,N))
    #for each city calculate distances to other cities
    #           note I could technically calculate only half then look up other half to speed up for large values of N
    for i in range(N):
        for j in range(N):
            D[i,j] = math.sqrt( (coords[i,0]-coords[j,0])**2 + (coords[i,1]-coords[j,1])**2) #calculate distance between points
            
            #write to file
            if (j == N-1):
                dist.write(str(D[i,j]) + "\n")
            else:
                dist.write(str(D[i,j]) + ",")
    dist.close()



######### Uncomment below as needed ###################
def main():
    
    # parse command line
    parser = argparse.ArgumentParser(description="Generates coordinates for cities and distances between them for TSP")
    parser.add_argument('coords_type',type=str,help='The types of coordinates generated: rand - random coordinates, circ - circle coordinates',default='rand')
    parser.add_argument('num_cities',type=int,help='The number of cities to generate',default=1000)
    parser.add_argument('out_file_name_prefix',type=str,help='Prefix file name to output to. (ex. "outfile" will result in distance matrix output of "outfile_dist.csv" and coordinate file of "outfile_coords.csv"',default=f'TSP')

    
    args = parser.parse_args()
    c_type = args.coords_type
    N = args.num_cities
    outfile = args.out_file_name_prefix
    
    # generate files
    coords_filename = f"{outfile}_coords.csv"
    dist_filename = f"{outfile}_dist.csv"
    
    
    if c_type == 'rand':
        GenerateCoords(coords_filename,N)
    elif c_type == 'circ':
        GenerateCircleCoords(coords_filename,N)
    else:
        ValueError("Incorrect Coords_type, must be 'rand' or 'circ'")
    CalcDistances(coords_filename,dist_filename)
    
if __name__ == '__main__':
    main()
    
    
    
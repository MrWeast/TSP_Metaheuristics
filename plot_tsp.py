import numpy as np
import matplotlib.pyplot as plt


def plot_tsp_solution(coords_file, solution):
    # Extract the coordinates based on the permutation
    permutation = solution.tolist()
    print(permutation)
    coords = np.genfromtxt(coords_file, delimiter=',')
    x = [coords[i,0] for i in permutation]
    y = [coords[i,1] for i in permutation]

    # Plot the TSP solution
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    
    
    # Annotate the cities
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.annotate(i, (xi, yi), textcoords="offset points", xytext=(-8,0), ha='center')
    
    plt.title('Traveling Salesman Problem Solution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()
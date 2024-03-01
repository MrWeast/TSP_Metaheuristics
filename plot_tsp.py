import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

def plot_tsp_solution(coords_file, solution):
    # Extract the coordinates based on the permutation
    permutation = solution.tolist()
    coords = np.genfromtxt(coords_file, delimiter=',')
    x = [coords[i,0] for i in permutation]
    y = [coords[i,1] for i in permutation]

    # Plot the TSP solution
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    
    
    # Annotate the cities
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.annotate(i, (xi, yi), textcoords="offset points", xytext=(-8,0), ha='center')
    
    plt.title('Traveling Salesman Problem Tour')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()
    



 

def update(frame, ax, coords, tsp_tours,itrs):
    ax.clear()
    tour = tsp_tours[frame]
    permutation = tour.tolist()
    x = [coords[i,0] for i in permutation]
    y = [coords[i,1] for i in permutation]
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    ax.plot(x, y, marker='o', linestyle='-', color='b')
    # Annotate the cities
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(i, (xi, yi), textcoords="offset points", xytext=(-8,0), ha='center')
    
    ax.set_title(f'Tour {frame}, found on iteration {itrs[frame-1]}')
    

def animated_plot(coords_file, tsp_tours,itrs, ani_name):
    fig, ax = plt.subplots()
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = ax.plot([], [], 'bo')
    # Set the axis limits based on city coordinates
    coords = np.genfromtxt(coords_file, delimiter=',')

    ani = FuncAnimation(fig, update, fargs=(ax, coords, tsp_tours,itrs), frames=len(tsp_tours), interval=100, repeat=False)
    ani.save(ani_name,writer='ffmpeg')
    return ani
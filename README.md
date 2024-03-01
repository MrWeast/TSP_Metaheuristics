dependencies:
- numpy
- matplotlib
- ffmpeg (only for animated plots)

Metaheuristics Used:
- Variable Neighborhood Search (VNS)
- Genetic Algorithm (GA)
- Local Search (LS)
- Simulated Annealing (SA)
- Particle Swarm (PS)

Includes:
- demonstration of the various metaheuristics in Metaheuristics.ipynb as well as the videos showing the evolution of found tours
- a way to generate new sets of cities with GenerateData.py
- plotting tours (animated and unanimated) in plot_tsp.py

Description:

- Traveling Salesman Problem: 
  - Given a set of cities that satisfy the triangle inequality, find the shortest path to visit each city without backtracking
  - I do not assume returning to starting city or specify a starting city, but the code provided can easily be adapted to make these assumptions


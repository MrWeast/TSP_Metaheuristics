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
  - to prevent infeasible solutions from being generated, PS operates on a random keys approach. These keys then sort an initial tour which results in a new tour. This gives PS a disadvange compared to other the metaheuristics since many keys can lead to the same sorting. However, PS is generally not well suited to discrete optimization problems for this reason.

Includes:
- Examples:
 - sample usage in Metaheuristics.ipynb, each metaheuristic is given the same number of objective function evaluations
 - distance and coordinate files used for sample usage
 - animated plots for each metaheuristics' evolution of best found solution
- a way to generate new sets of cities with GenerateData.py
 - can generate cities arranged along a circle or randomly placed (see file for useage)
- plotting tours (animated and unanimated) in plot_tsp.py

Description:
- Traveling Salesman Problem: 
  - Given a set of cities that satisfy the triangle inequality, find the shortest path to visit each city without backtracking
  - I do not assume returning to starting city or specify a starting city, but the code provided can easily be adapted to make these assumptions

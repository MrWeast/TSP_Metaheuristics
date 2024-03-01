import numpy as np
#import matplotlib.pyplot as plt
import random as rand
import math



#given an array calculates sum of distances
def evaluate(D,tour):
    # tour, tour to evaluate
    # D, distance matrix
    tour = tour.astype(int)
    val = 0
    for i in range(tour.shape[0]-1):
        val += D[tour[i], tour[i+1]]
    return val


def EstimateLowerBound(D):
    # Estimate of the lower bound of the best tour. 
    # Start at city 0 and subsequently move to the closest city that has not yet been visited.
    n = D.shape[0] 
    tour = np.zeros(n, dtype=int)
    unvisited_cities = set(range(1, n))

    for i in range(1, n):
        closest_cities = list(unvisited_cities)
        closest_cities.sort(key=lambda city: D[tour[i - 1], city])
        
        next_city = closest_cities[0]
        tour[i] = next_city
        unvisited_cities.remove(next_city)


    value = evaluate(D, tour)
    
    return tour, value


def swap(tour):
    #generates two random indicies and swaps the values at them
    indices = np.random.permutation(tour.shape[0])[:2]
    tour[indices[0]], tour[indices[1]] = tour[indices[1]], tour[indices[0]]

    return tour


def pairwise_swap(tour):
    #pairwise swap, swap order of two cities that are next to each other in the tour
    
    # generate index
    x1 = rand.randint(0,tour.shape[0]-1)

    if (x1 == tour.shape[0]-1): # if last city is selected then swap with previous city
        x2 = x1-1
    else: # otherwise swap with subsequent city
        x2 = x1+1

    #swap values at indexes
    v1 = tour[x1]
    v2 = tour[x2]
    
    tour[x1] = v2
    tour[x2] = v1

    return tour


def multi_pair_swap(tour):
    #random pair swap, generates several random pairs and swaps them
    indices = np.random.permutation(tour.shape[0])[:6]
    
    #swap values at indexes
    tour[indices[0]], tour[indices[1]] = tour[indices[1]], tour[indices[0]]
    tour[indices[2]], tour[indices[3]] = tour[indices[3]], tour[indices[2]]
    tour[indices[4]], tour[indices[5]] = tour[indices[5]], tour[indices[4]]

    return tour


def inversion(tour):
    #inversion swap, inverts sequence between two indicies
    indices = np.random.permutation(tour.shape[0])[:2]
    #generate indicies
    x1 = indices[0]
    x2 = indices[1]

    #need to make sure the smaller index is in the first posistion for inversion
    if (x1 < x2):
        tour[x1:x2] = tour[x1:x2][::-1]
    else:
        tour[x2:x1] = tour[x2:x1][::-1]

    return tour


#Neighborhood swap method
def Var_Neighborhood_Swap(tour,k):
    # k is neighbordhood type
    if k == 1:
        tour = pairwise_swap(tour)
    elif k==2:
        tour = swap(tour)
    elif k==3:
        tour = multi_pair_swap(tour)
    elif k==4:
        tour = inversion(tour)
    else:
        KeyError("k should not be greater than 4")
        
    return tour




#generates points for particle swarms and initalizes velocities to 0
def Genrate_Points(D,num_points):
    #create an array with each row representing a particle, w/ random keys (intigers b/w 0 and 10,000)
    Particles = np.random.randint(D.shape[1],size=(num_points,D.shape[1]))
    Velocity = np.random.randint(-D.shape[1], D.shape[1],size=(num_points,D.shape[1]))
    return Particles, Velocity


#evaluates objective function value of all particle in particle swarm and returns an array of these values and the index of the best one
def Evaluate_Particles(Particles,initial_tour,D):
    particle_evaluations = np.zeros(Particles.shape[0]) #list of tour distances based on particles (keys)
    temp_array = np.zeros(shape = (2,Particles.shape[1])) #top row is actual tour to be sorted, bottom row is the key to sort by
    for i in range(Particles.shape[0]):
        temp_array[0,:] = initial_tour #tour to be sorted based on key
        temp_array[1,:] = Particles[i,:] #key to sort by
        temp_tour = temp_array[1,:].argsort().astype(int) #temp tour now sorted by key
        particle_evaluations[i] = evaluate(D,temp_tour) #evaluate how each tour did
        best_tour_index = np.argmin(particle_evaluations) #index of best tour

    return best_tour_index,particle_evaluations

#updates velocities of all particles in particle swarm
def Update_Velocity(weights,Particles,Velocity,Global_Best_Paticle,Current_Particle_Best):
    for i in range(Velocity.shape[0]):
        Velocity[i,:] = weights[0]*Velocity[i,:] + \
                        weights[1]*rand.uniform(0,1)*(Current_Particle_Best[i,:] - Particles[i,:]) + \
                        weights[2]*rand.uniform(0,1)*(Global_Best_Paticle - Particles[i,:])
    return Velocity


def Generate_Chromosomes(num_chromosomes,num_cities):
    # returns a population of size (num_chromosomes, num_cities)
    population = np.zeros((num_chromosomes//2,num_cities),dtype=int)
    tour = np.arange(num_cities)

    for i in range(0,population.shape[0]):
        np.random.shuffle(tour)
        population[i,:] = tour
        
    return (population)

def Cross_Over(population):
    numParents = population.shape[0]
    newPop = np.zeros((2*numParents,population.shape[1])) #where children are placed
    
    #generate new population by selecting random parents and doing random index crossover
    for i in range(0,2*numParents,2):
        x1 = rand.randint(0,population.shape[0]-1) #index of first parent
        x2 = rand.randint(0,population.shape[0]-1) #index of second parent
        while x1==x2: #ensure non duplicate index
            x2 = rand.randint(0,population.shape[0]-1)
        
        #cross over index
        cross_over_idx = rand.randint(0,population.shape[1])
        
        # cross over operation for child 1
        p1_elements = population[x1,:cross_over_idx]
        p2_elements = [element for element in population[x2] if element not in p1_elements]
        
        tour = np.concatenate((p1_elements, p2_elements))
        newPop[i] = tour
        
        # cross over operation for child 2
        p1_elements = population[x2,:cross_over_idx]
        p2_elements = [element for element in population[x1] if element not in p1_elements]
        
        tour = np.concatenate((p1_elements, p2_elements))
        newPop[i+1] = tour
        
        
        

    return(newPop)
        
        
def Mutation(population):
    numMutate = rand.randint(1,int(population.shape[0])*.1) #random number of children to mutate, capped at 10% of population
    
    for i in range(numMutate):
        x1 = rand.randint(0,population.shape[0]-1) #index of child to mutate
        population[x1,:] = inversion(population[x1,:])
    return population
        
    
def Evaluate_population(D,population):
    '''
    Inputs:
        population: the current population of tours
        D: the distance matrix
    
    '''
    population_evaluations = np.zeros(population.shape[0]) #list of tour distances
    for i in range(population.shape[0]):
        population_evaluations[i] = evaluate(D,population[i]) #evaluate how each tour did
    best_tour_index = np.argmin(population_evaluations) #index of best tour

    return best_tour_index, population_evaluations


#prints info an creates copy of inital tour info to initalize current best solutions
def print_init_info(D,initial_tour, k):
    
    print("\n-----------------------------------------------------------")
    if k ==1:
        print("Local Search")
    if k==2:
        print("Simulated Annealing")
    if k==3:
        print("Variable Neighborhood Search")
    if k==4:
        print("Particle Swarm")
    if k==5:
        print("Gentic Algorithm")
    print("-----------------------------------------------------------\n")
    print("The inital tour and distance is: \n")
    best_tour = initial_tour.copy()
    best_solution = evaluate(D,best_tour)
    print(best_tour)
    print(best_solution)
    print("\n-----------------------------------------------------------\n")
    return best_tour, best_solution


#local search that moves to new tour as soon as a better tour is found
def Local_Search(D,inital_tour, count_limit):
    
    #print inital info and initialize solution
    best_tour, best_solution = print_init_info(D,inital_tour,1)

    #initialize stuff before loop
    count = 0 #count of iterations
    curr_tour = best_tour.copy()
    curr_solution = best_solution
    isRunning = True

    is_better_count = 0
    while(isRunning):
        #generate new tour
        curr_tour = swap(best_tour.copy())
        curr_solution = evaluate(D,curr_tour)

        #if new tour is better than current best then update best tour info
        if (curr_solution < best_solution): 
            best_tour = curr_tour.copy()
            best_solution = curr_solution
            is_better_count +=1

        count += 1
        if (count > count_limit): #if termination conditions are met then stop
            isRunning = False
    
    #report findings
    print(f"{is_better_count} swaps yielded a better solution\n")
    print(f"{count} swaps total\n")
    print("The final tour for local search is: \n")
    print(best_tour)
    print(f"The final tour distance for local search is: {best_solution}\n")
    print("\n-----------------------------------------------------------\n")
    return (best_tour, best_solution)

def Simulated_Annealing(D,initial_tour, count_limit,Temp):
   
    #print inital info and initialize solution
    best_tour, best_solution = print_init_info(D,initial_tour,2)

    #initalize stuff before loop
    count = 0 #count of iterations
    curr_tour = best_tour.copy()
    curr_solution = best_solution
    isRunning = True
    T = Temp #the temperature

    tmp_tour = curr_tour.copy
    tmp_solution = curr_solution

    while(isRunning):
        #reset inner loop variables
        isWorse = True
        while(isWorse):
            #generate tour based on current tour
            tmp_tour = swap(curr_tour)
            tmp_solution = evaluate(D,tmp_tour)
            #if better than current solution then make it the new optimal and set it to current tour
            if (tmp_solution < best_solution):
                best_tour = tmp_tour.copy()
                best_solution = tmp_solution

                curr_tour = tmp_tour
                curr_solution = tmp_solution
                isWorse = False
            #if it is worse than the current tour then randomly move to it
            elif (rand.random() < min(1, math.e**(-(tmp_solution-curr_solution)/T))):
                curr_tour = tmp_tour
                curr_solution = tmp_solution
                isWorse = False
            #if fail limit reached then stop
            if count > count_limit: break
            count +=1
        T *=0.9 # update temp
        #if stopping conditions met then stop
        if (count > count_limit):
            isRunning = False
    
    #report findings
    print("The final tour for Simulated Annealing is: \n")
    print(best_tour)
    print(f"The final distance for Simulated Annealing is: {best_solution}\n")
    print("\n------------------------------------------------------------------\n")

    return (best_tour, best_solution)


#local search that moves to new tour as soon as a better tour is found
def Variable_Neighborhood_Search(D,initial_tour,fail_limit, count_limit):
    
    #print inital info and initialize solution
    best_tour, best_solution = print_init_info(D,initial_tour,3)


    #initialize stuff before loop
    count = 0 #count of iterations
    fail_count = 0 #number of failed attempts at finding a new solution
    curr_tour = best_tour.copy()
    curr_solution = best_solution
    isRunning = True
    k = 1


    size = np.size(best_tour)

    while(isRunning):
        #reset isWorse and fail_count after a successful swap
        isWorse = True
        fail_count = 0
        while(isWorse):
            curr_tour = Var_Neighborhood_Swap(best_tour.copy(),k) #generate a new tour
            curr_solution = evaluate(D,curr_tour) #evaluate a new tour

            unique_elements, counts = np.unique(curr_tour, return_counts=True)
            if np.size(unique_elements) < size:
                print("The array has duplicates.")
                print('k',k)
                exit



            if (curr_solution < best_solution): # if new tour is better than current best update best_tour and exit inner loop
                best_tour = curr_tour
                best_solution = curr_solution
                isWorse = False
            elif (fail_count > fail_limit): #if fail limit is reached then exit loop
                isWorse = False
            else: #otherwise update fail count
                fail_count +=1

            if (fail_count >= fail_limit): #if fail count is half the limit and we have not gotten to the last neighborhood then update to next neighborhood and try again
                k+=1
                isWorse = False
                if k > 4: 
                    k=1 # reset neighborhood if looped through them all
            
            count += 1 #total number of obj function evaluations
            if (count > count_limit): break
        if (count > count_limit):
            isRunning = False
    
    #report findings
    print(str(count) + " objective evaluations occured \n")
    print("The final tour and distance for Variable Neighborhood Search is: \n")
    print(best_tour)
    print(best_solution)
    print("\n-----------------------------------------------------------\n")
    return (best_tour, best_solution)


def Particle_Swarm(D,count_limit, weights, num_particles,initial_tour):
    #print intial info
    best_tour, best_solution = print_init_info(D,initial_tour,4)

    #generate inital random keys
    Particles,Velocity = Genrate_Points(D,num_particles)
    Current_Particle_Best = Particles.copy() #current best key for each particle

    #evaluate these particles
    best_key_index,particle_evaluations = Evaluate_Particles(Particles,initial_tour,D)

    #set best key and best value
    Best_Key = Particles[best_key_index,:].copy()
    Best_Value = particle_evaluations[best_key_index]
    Current_Particle_Best_Values = particle_evaluations

    for i in range(count_limit):
        #update velocity of each particle
        Velocity = Update_Velocity(weights,Particles, Velocity,Best_Key, Current_Particle_Best)
        #update each particle posistion
        Particles = Particles + Velocity
        #evaluate each particle
        curr_eval_best_index,particle_evaluations = Evaluate_Particles(Particles,initial_tour,D)
        #update global best if a new evaluation is better than the current best
        if (particle_evaluations[curr_eval_best_index] < Best_Value):
            best_key_index = curr_eval_best_index #update best index
            Best_Key = Particles[best_key_index] #update best key
            Best_Value = particle_evaluations[best_key_index] #update best value
        
        #update each particle's personal best
        for j in range(Particles.shape[0]):
            #if a current particle's evaluation is better than its current best
            if (particle_evaluations[j] < Current_Particle_Best_Values[j]):
                #update that particle's current best key
                Current_Particle_Best[j,:] = Particles[j,:]


    temp_array = np.zeros((2,Particles.shape[1]))
    temp_array[0,:] = initial_tour #tour to be sorted based on key
    temp_array[1,:] =Best_Key #key to sort by
    best_tour = temp_array[1,:].argsort().astype(int) #temp tour now sorted by key
    best_solution = evaluate(D,best_tour)
    

    print(str(count_limit) + " iterations were performed\n")
    print("With " + str(num_particles) + " number of particles\n")
    print("The final tour and distance for Particle Swarm is: \n")
    print(best_tour)
    print(best_solution)
    print("\n-----------------------------------------------------------\n")


    return (best_tour, best_solution)

def GA(D,count_limit, num_chromosomes):
    
    print("\n-----------------------------------------------------------\n")
    print("Genetic Algorithm")
    print("\n-----------------------------------------------------------\n")
    #generate inital random keys
    population = Generate_Chromosomes(num_chromosomes,D.shape[1])


    #evaluate these parent
    best_tour_index, population_evaluations = Evaluate_population(D,population)

    #set best key and best value
    best_tour = population[best_tour_index,:].copy()
    best_value = population_evaluations[best_tour_index]
    
    
    for i in range(count_limit):
        
        #perform crossover
        population = Cross_Over(population)
        #do mutation
        population = Mutation(population)
        #find best solution
        best_tour_index, population_evaluations = Evaluate_population(D,population)
        #update best tour and best value
        if population_evaluations[best_tour_index] < best_value:
            best_tour = population[best_tour_index].copy()
            best_value = population_evaluations[best_tour_index]
            
        # cut population down to top half of tours
        sort = population_evaluations.argsort()
        sort = sort[:sort.shape[0]//2]
        population = population[sort]
        
        
    #report solution
    
    print(f"{count_limit} iterations were performed\n")
    print(f"With population size of {num_chromosomes}\n")
    print("The final tour for GA is: \n")
    print(best_tour)
    print(f"The final distance for GA is: {best_value}\n")
    
    return best_tour.astype(int), best_value







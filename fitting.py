import numpy as np
import main as ma
import constants as const
from multiprocessing import Pool
from scipy.optimize import curve_fit
import scipy as sc
#from numba import jit

# Data from https://www.n-tv.de/infografik/Coronavirus-aktuelle-Zahlen-Daten-zur-Epidemie-in-Deutschland-Europa-und-der-Welt-article21604983.html
# from 08.03.2020 to 13.04.2020
number_sick_original = [1035, 1180, 1563, 1899, 2746, 3675, 4599, 5796, 7232, 9375, 12300, 15305, 19655, 22189, 21963, 26102, 30014, 34252, 37301, 42968, 48241, 52584, 48469, 50271, 53389, 57048, 60483, 63595, 65798, 66767, 67922, 59823, 60326, 61340, 61235, 61208, 59322, ]
number_immune_original = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2809, 2809, 2809, 2809, 5600, 5600, 5600, 5600, 13500, 16100, 18700, 21400, 23800, 26400, 28700, 30600, 33300, 46300, 50000, 53913, 57400, 60200, 64300]
population_germany = 80000000
faktor_actual_cases = 100


number_sick = np.asarray(number_sick_original) * faktor_actual_cases
number_immune = np.asarray(number_immune_original) * 10

number_sick = number_sick / float(population_germany) * const.numPop

print(number_sick)
print(len(number_sick))

n = number_sick[0] / float(population_germany)
print(n)


def simulate_multi(x, U, t, V):
    number_processes = const.number_processes
    numberDays = 34
    n = 3.234375e-07

    poolarray = []
    for i in range(number_processes):
        poolarray.append((U, t, V, n, numberDays, 42 + i))
    pool = Pool(processes=number_processes)
    result = pool.starmap(ma.simulation, poolarray)

    arrayInfected = np.asarray(result[0])
    for i in range(1, number_processes):
        arrayInfected += np.asarray(result[i])
        # print(result[i])
    arrayInfected = arrayInfected / float(number_processes)

    return arrayInfected[x]

def simulate_multi2(U, t, V):
    number_processes = const.number_processes
    numberDays = 34
    n = 3.234375e-07

    poolarray = []
    for i in range(number_processes):
        poolarray.append((U, t, V, n, numberDays, 42 + i))
    with Pool(processes=number_processes) as pool:
        result = pool.starmap(ma.simulation, poolarray)
    #pool = Pool(processes=number_processes)
    #result = pool.starmap(ma.simulation, poolarray)

    arrayInfected = np.asarray(result[0])
    for i in range(1, number_processes):
        arrayInfected += np.asarray(result[i])
        # print(result[i])
    arrayInfected = arrayInfected / float(number_processes)

    return arrayInfected

def simulate(x, U, t, V):
    number_processes = const.number_processes
    number_loops = number_processes
    numberDays = 34
    n = 3.234375e-07

    arrayInfected = ma.simulation(U=U, t=t, V=V, n=n, numberDays=numberDays, seed=42)
    arrayInfected = np.asarray(arrayInfected)
    for i in range(1, number_loops):
        arrayInfected += np.asarray(ma.simulation(U=U, t=t, V=V, n=n, numberDays=numberDays, seed=42))

    arrayInfected = arrayInfected / float(number_loops)

    return arrayInfected[x]


def fitting_scipy(xdata, ydata):
    popt, pcov = curve_fit(simulate_multi, xdata, ydata, bounds=([0.5, 0.00001, 0.000001], [0.85, 0.04, 0.005]))
    print(popt)
    print(pcov)
    params = ['U', 't', 'V']
    with open('parameter_scipy.txt', 'w') as f:
        for i in range(len(popt)):
            f.write(params[i] + ': ' + str(popt[i]) + '\n')


print(sc.__version__)
xdata = range(0, 34)
print(xdata)
#fitting_scipy(xdata=xdata, ydata=number_sick)


def fitting_genetic(actual_number_sick, populationsize = 30, number_generations = 10):
    def crossover(parent1, parent2):
        child1 = []
        child2 = []
        for i in range(len(parent1)):
            r = np.random.rand()
            if(r>0.5):
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return child1, child2

    def selection(population):
        mating_pool = []
        for i in range(len(population)):
            index1 = np.random.randint(low=0, high=len(population))
            index2 = np.random.randint(low=0, high=len(population))
            while(index1 == index2):
                index2 = np.random.randint(low=0, high=len(population))
            if(population[index1][3] < population[index2][3]):
                mating_pool.append(population[index1])
            else:
                mating_pool.append(population[index2])
        return mating_pool

    def mutation(individual, U_max, U_min, t_max, t_min, V_max, V_min):
        #how big is the mutation
        step_percent = 0.1
        gen_index = np.random.randint(low=0, high=3)
        tmp = [U_max, U_min, t_max, t_min, V_max, V_min]
        mut = 2 * (np.random.rand() - 0.5) * (tmp[2 * gen_index] - tmp[2 * gen_index + 1]) * step_percent
        #ensure, that the values are in the alllowed area
        while(individual[gen_index] + mut < tmp[2*gen_index + 1] or individual[gen_index] + mut > tmp[2*gen_index]):
            mut = 2 * (np.random.rand() - 0.5) * (tmp[2 * gen_index] - tmp[2 * gen_index + 1]) * step_percent
        individual[gen_index] += mut
        return individual

    U_min = 0.5
    U_max = 0.95
    t_min = 0.0001
    t_max = 0.15
    V_min = 0.00001
    V_max = 0.01
    probability_mutation = 0.2
    #initialize population
    population = []
    for i in range(populationsize):
        #between 0.5 and 0.9530
        U = np.random.rand() * (U_max - U_min) + U_min
        #between 0.0001 and 0.15
        t = np.random.rand() * (t_max - t_min) + t_min
        #between 0.00001 and 0.01
        V = np.random.rand() * (V_max - V_min) + V_min

        result = simulate_multi2(U, t, V)
        #print(result)
        #print(actual_number_sick)
        fitness = np.linalg.norm(actual_number_sick - result)

        ind = [U, t, V, fitness]
        population.append(ind)

    #find best individual
    #TODO check if I need deepcopy
    best_ind = population[0].copy()
    best_ind_fitness = best_ind[3]
    #print(population)
    #print(best_ind)
    #print(best_ind_fitness)
    for i in range(1, len(population)):
        if(population[i][3] < best_ind_fitness):
            # TODO check if I need deepcopy
            best_ind = population[i].copy()
            best_ind_fitness = best_ind[3]
            print('Fitness: ' + str(best_ind_fitness))

    #create new generations
    for gen in range(1, number_generations + 1):
        print('Generation: ' + str(gen))
        mating_pool = selection(population)
        new_population = list()
        for i in range(0, len(mating_pool), 2):
            parent1 = mating_pool[i]
            if(i+1 < len(mating_pool)):
                parent2 = mating_pool[i+1]
            else:
                parent2 = mating_pool[0]
            child1, child2 = crossover(parent1, parent2)
            if(np.random.rand() < probability_mutation):
                child1 = mutation(child1, U_max, U_min, t_max, t_min, V_max, V_min)
                print('mutation 1')
            if (np.random.rand() < probability_mutation):
                child2 = mutation(child2, U_max, U_min, t_max, t_min, V_max, V_min)
                print('mutation 2')
            result1 = simulate_multi2(child1[0], child1[1], child1[2])
            result2 = simulate_multi2(child2[0], child2[1], child2[2])
            fitness1 = np.linalg.norm(actual_number_sick - result1)
            fitness2 = np.linalg.norm(actual_number_sick - result2)
            child1[3] = fitness1
            child2[3] = fitness2
            new_population.append(child1)
            new_population.append(child2)
        population = new_population.copy()

        #search, if there is a new best individual
        for i in range(1, len(population)):
            if (population[i][3] < best_ind_fitness):
                # TODO check if I need deepcopy
                best_ind = population[i].copy()
                best_ind_fitness = best_ind[3]
                print('found new best individual')
                print('Fitness: ' + str(best_ind_fitness))

    #save result:
    params = ['U', 't', 'V', 'Fitness']
    with open('parameter_genetic.txt', 'w') as f:
        for i in range(len(best_ind)):
            f.write(params[i] + ': ' + str(best_ind[i]) + '\n')




fitting_genetic(actual_number_sick=number_sick)
















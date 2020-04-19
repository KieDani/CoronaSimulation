import numpy as np
import main as ma
import constants as const
from multiprocessing import Pool
from scipy.optimize import curve_fit
import scipy as sc
#from numba import jit

number_sick = const.number_sick
number_immune = const.number_immune


#I guess till 28.03.2020
number_sick_before = number_sick[0:const.time_lockdown]
#after 28.03.2020
number_sick_after = number_sick[const.time_lockdown:]


print(number_sick)
print(len(number_sick))


def simulate_multi_scipy(x, U, t, V):
    U_array = list()
    t_array = list()
    V_array = list()
    for i in range(len(x)):
        U_array.append(U)
        t_array.append(t)
        V_array.append(V)

    number_processes = const.number_processes
    numberDays = len(U)
    n = const.n

    poolarray = []
    for i in range(number_processes):
        poolarray.append((U_array, t_array, V_array, n, numberDays, 42 + i))
    pool = Pool(processes=number_processes)
    result = pool.starmap(ma.simulation, poolarray)

    arrayInfected = np.asarray(result[0][0])
    for i in range(1, number_processes):
        arrayInfected += np.asarray(result[i][0])
        # print(result[i])
    arrayInfected = arrayInfected / float(number_processes)

    return arrayInfected[x]

def simulate_multi_genetic(U, t, V, numberOfDays):
    number_processes = const.number_processes
    numberDays = numberOfDays
    n = const.n

    poolarray = []
    for i in range(number_processes):
        poolarray.append((U, t, V, n, numberDays, 42 + i))
    with Pool(processes=number_processes) as pool:
        result = pool.starmap(ma.simulation, poolarray)
    #pool = Pool(processes=number_processes)
    #result = pool.starmap(ma.simulation, poolarray)

    arrayInfected = np.asarray(result[0][0])
    for i in range(1, number_processes):
        arrayInfected += np.asarray(result[i][0])
        # print(result[i])
    arrayInfected = arrayInfected / float(number_processes)

    return arrayInfected


def fitting_scipy(xdata, ydata):
    popt, pcov = curve_fit(simulate_multi_scipy, xdata, ydata, bounds=([0.5, 0.00001, 0.000001], [0.85, 0.04, 0.005]))
    print(popt)
    print(pcov)
    params = ['U', 't', 'V']
    with open('parameter_scipy.txt', 'w') as f:
        for i in range(len(popt)):
            f.write(params[i] + ': ' + str(popt[i]) + '\n')


#print(sc.__version__)
xdata = range(0, 34)
#print(xdata)
#fitting_scipy(xdata=xdata, ydata=number_sick)


#does not allow a lockdown
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

        U_array = list()
        t_array = list()
        V_array = list()
        for i in range(len(actual_number_sick)):
            U_array.append(U)
            t_array.append(t)
            V_array.append(V)
        result = simulate_multi_genetic(U_array, t_array, V_array, numberOfDays = len(actual_number_sick))
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

            U_array1 = list()
            t_array1 = list()
            V_array1 = list()
            U_array2 = list()
            t_array2 = list()
            V_array2 = list()
            for i in range(len(actual_number_sick)):
                U_array1.append(child1[0])
                t_array1.append(child1[1])
                V_array1.append(child1[2])
                U_array2.append(child2[0])
                t_array2.append(child2[1])
                V_array2.append(child2[2])
            result1 = simulate_multi_genetic(U_array1, t_array1, V_array1, numberOfDays=len(actual_number_sick))
            result2 = simulate_multi_genetic(U_array2, t_array2, V_array2, numberOfDays=len(actual_number_sick))
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
    with open('parameter_genetic_tmp.txt', 'w') as f:
        for i in range(len(best_ind)):
            f.write(params[i] + ': ' + str(best_ind[i]) + '\n')

    return best_ind


#allows for a change of parameters because of a lockdown
def fitting_genetic_lockdown(actual_number_sick, populationsize = 10, number_generations = 3, time_lockdown = const.time_lockdown):
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
                index2 = np.random.randint(low=0, high=len(population)-1)
            if(population[index1][6] < population[index2][6]):
                mating_pool.append(population[index1])
            else:
                mating_pool.append(population[index2])
        return mating_pool

    def mutation(individual, U_max, U_min, t_max, t_min, V_max, V_min):
        #how big is the mutation
        step_percent = 0.1
        gen_index = np.random.randint(low=0, high=len(individual) - 1)
        print(gen_index)
        tmp = [U_max, U_min, t_max, t_min, V_max, V_min, U_max, U_min, t_max, t_min, V_max, V_min]
        mut = 2 * (np.random.rand() - 0.5) * (tmp[2 * gen_index] - tmp[2 * gen_index + 1]) * step_percent
        #ensure, that the values are in the alllowed area
        while(individual[gen_index] + mut < tmp[2*gen_index + 1] or individual[gen_index] + mut > tmp[2*gen_index]):
            mut = 2 * (np.random.rand() - 0.5) * (tmp[2 * gen_index] - tmp[2 * gen_index + 1]) * step_percent
        individual[gen_index] += mut
        return individual

    U_min = 0.5
    U_max = 0.95
    t_min = 0.0001
    t_max = 0.25
    V_min = 0.00001
    V_max = 0.01
    probability_mutation = 0.2
    #initialize population
    population = []
    for i in range(populationsize):
        #between 0.5 and 0.9530
        U_before = np.random.rand() * (U_max - U_min) + U_min
        U_after = np.random.rand() * (U_max - U_min) + U_min
        #between 0.0001 and 0.15
        t_before = np.random.rand() * (t_max - t_min) + t_min
        t_after = np.random.rand() * (t_max - t_min) + t_min
        #between 0.00001 and 0.01
        V_before = np.random.rand() * (V_max - V_min) + V_min
        V_after = np.random.rand() * (V_max - V_min) + V_min

        U_array = list()
        t_array = list()
        V_array = list()
        for j in range(len(actual_number_sick[:time_lockdown])):
            U_array.append(U_before)
            t_array.append(t_before)
            V_array.append(V_before)
        for j in range(len(actual_number_sick[time_lockdown:])):
            U_array.append(U_after)
            t_array.append(t_after)
            V_array.append(V_after)
        result = simulate_multi_genetic(U_array, t_array, V_array, numberOfDays = len(actual_number_sick))
        #print(result)
        #print(actual_number_sick)
        fitness = np.linalg.norm(actual_number_sick - result)

        ind = [U_before, t_before, V_before, U_after, t_after, V_after, fitness]
        population.append(ind)

    #find best individual
    #TODO check if I need deepcopy
    best_ind = population[0].copy()
    best_ind_fitness = best_ind[6]
    #print(population)
    #print(best_ind)
    #print(best_ind_fitness)
    for i in range(1, len(population)):
        if(population[i][6] < best_ind_fitness):
            # TODO check if I need deepcopy
            best_ind = population[i].copy()
            best_ind_fitness = best_ind[6]
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

            U_array1 = list()
            t_array1 = list()
            V_array1 = list()
            U_array2 = list()
            t_array2 = list()
            V_array2 = list()
            for j in range(len(actual_number_sick[:time_lockdown])):
                U_array1.append(child1[0])
                t_array1.append(child1[1])
                V_array1.append(child1[2])
                U_array2.append(child2[0])
                t_array2.append(child2[1])
                V_array2.append(child2[2])
            for j in range(len(actual_number_sick[time_lockdown:])):
                U_array1.append(child1[3])
                t_array1.append(child1[4])
                V_array1.append(child1[5])
                U_array2.append(child2[3])
                t_array2.append(child2[4])
                V_array2.append(child2[5])
            result1 = simulate_multi_genetic(U_array1, t_array1, V_array1, numberOfDays=len(actual_number_sick))
            result2 = simulate_multi_genetic(U_array2, t_array2, V_array2, numberOfDays=len(actual_number_sick))
            fitness1 = np.linalg.norm(actual_number_sick - result1)
            fitness2 = np.linalg.norm(actual_number_sick - result2)
            child1[6] = fitness1
            child2[6] = fitness2
            new_population.append(child1)
            new_population.append(child2)
        population = new_population.copy()

        #search, if there is a new best individual
        for i in range(1, len(population)):
            if (population[i][6] < best_ind_fitness):
                # TODO check if I need deepcopy
                best_ind = population[i].copy()
                best_ind_fitness = best_ind[6]
                print('found new best individual')
                print('Fitness: ' + str(best_ind_fitness))

    #save result:
    params = ['U_before', 't_before', 'V_before', 'U_after', 't_after', 'V_after', 'Fitness']
    with open('parameter_genetic_tmp.txt', 'w') as f:
        for i in range(len(best_ind)):
            f.write(params[i] + ': ' + str(best_ind[i]) + '\n')

    return best_ind

# keeps the 10% best individuals for the next generation
def fitting_genetic_lockdown_percent(actual_number_sick, populationsize=10, number_generations=3, time_lockdown=const.time_lockdown, percent = 0.05):
    def crossover(parent1, parent2):
        child1 = []
        child2 = []
        for i in range(len(parent1)):
            r = np.random.rand()
            if (r > 0.5):
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
            while (index1 == index2):
                index2 = np.random.randint(low=0, high=len(population) - 1)
            if (population[index1][6] < population[index2][6]):
                mating_pool.append(population[index1])
            else:
                mating_pool.append(population[index2])
        return mating_pool

    def mutation(individual, U_max, U_min, t_max, t_min, V_max, V_min):
        # how big is the mutation
        step_percent = 0.1
        gen_index = np.random.randint(low=0, high=len(individual) - 1)
        print(gen_index)
        tmp = [U_max, U_min, t_max, t_min, V_max, V_min, U_max, U_min, t_max, t_min, V_max, V_min]
        mut = 2 * (np.random.rand() - 0.5) * (tmp[2 * gen_index] - tmp[2 * gen_index + 1]) * step_percent
        # ensure, that the values are in the alllowed area
        while (individual[gen_index] + mut < tmp[2 * gen_index + 1] or individual[gen_index] + mut > tmp[
            2 * gen_index]):
            mut = 2 * (np.random.rand() - 0.5) * (tmp[2 * gen_index] - tmp[2 * gen_index + 1]) * step_percent
        individual[gen_index] += mut
        return individual

    def sort_population(population):
        for i in range(1, len(population)):
            to_be_sorted = population[i]
            j = i
            while (j > 0) and (population[j - 1][-1] > to_be_sorted[-1]):
                population[j] = population[j - 1]
                j -= 1
            population[j] = to_be_sorted
        return population



    U_min = 0.5
    U_max = 0.95
    t_min = 0.0001
    t_max = 0.25
    V_min = 0.00001
    V_max = 0.01
    probability_mutation = 0.2
    # initialize population
    population = []
    for i in range(populationsize):
        # between 0.5 and 0.9530
        U_before = np.random.rand() * (U_max - U_min) + U_min
        U_after = np.random.rand() * (U_max - U_min) + U_min
        # between 0.0001 and 0.15
        t_before = np.random.rand() * (t_max - t_min) + t_min
        t_after = np.random.rand() * (t_max - t_min) + t_min
        # between 0.00001 and 0.01
        V_before = np.random.rand() * (V_max - V_min) + V_min
        V_after = np.random.rand() * (V_max - V_min) + V_min

        U_array = list()
        t_array = list()
        V_array = list()
        for j in range(len(actual_number_sick[:time_lockdown])):
            U_array.append(U_before)
            t_array.append(t_before)
            V_array.append(V_before)
        for j in range(len(actual_number_sick[time_lockdown:])):
            U_array.append(U_after)
            t_array.append(t_after)
            V_array.append(V_after)
        result = simulate_multi_genetic(U_array, t_array, V_array, numberOfDays=len(actual_number_sick))
        # print(result)
        # print(actual_number_sick)
        fitness = np.linalg.norm(actual_number_sick - result)

        ind = [U_before, t_before, V_before, U_after, t_after, V_after, fitness]
        population.append(ind)

    # find best individual
    # TODO check if I need deepcopy
    best_ind = population[0].copy()
    best_ind_fitness = best_ind[6]
    # print(population)
    # print(best_ind)
    # print(best_ind_fitness)
    for i in range(1, len(population)):
        if (population[i][6] < best_ind_fitness):
            # TODO check if I need deepcopy
            best_ind = population[i].copy()
            best_ind_fitness = best_ind[6]
            print('Fitness: ' + str(best_ind_fitness))

    # create new generations
    for gen in range(1, number_generations + 1):
        print('Generation: ' + str(gen))
        mating_pool = selection(population)
        new_population = population.copy()
        #keep only best 10% of population
        new_population = sort_population(new_population)[:int(np.ceil(percent * populationsize))]
        for i in range(0, len(mating_pool), 2):
            parent1 = mating_pool[i]
            if (i + 1 < len(mating_pool)):
                parent2 = mating_pool[i + 1]
            else:
                parent2 = mating_pool[0]
            child1, child2 = crossover(parent1, parent2)
            if (np.random.rand() < probability_mutation):
                child1 = mutation(child1, U_max, U_min, t_max, t_min, V_max, V_min)
                print('mutation 1')
            if (np.random.rand() < probability_mutation):
                child2 = mutation(child2, U_max, U_min, t_max, t_min, V_max, V_min)
                print('mutation 2')

            U_array1 = list()
            t_array1 = list()
            V_array1 = list()
            U_array2 = list()
            t_array2 = list()
            V_array2 = list()
            for j in range(len(actual_number_sick[:time_lockdown])):
                U_array1.append(child1[0])
                t_array1.append(child1[1])
                V_array1.append(child1[2])
                U_array2.append(child2[0])
                t_array2.append(child2[1])
                V_array2.append(child2[2])
            for j in range(len(actual_number_sick[time_lockdown:])):
                U_array1.append(child1[3])
                t_array1.append(child1[4])
                V_array1.append(child1[5])
                U_array2.append(child2[3])
                t_array2.append(child2[4])
                V_array2.append(child2[5])
            result1 = simulate_multi_genetic(U_array1, t_array1, V_array1, numberOfDays=len(actual_number_sick))
            result2 = simulate_multi_genetic(U_array2, t_array2, V_array2, numberOfDays=len(actual_number_sick))
            fitness1 = np.linalg.norm(actual_number_sick - result1)
            fitness2 = np.linalg.norm(actual_number_sick - result2)
            child1[6] = fitness1
            child2[6] = fitness2


            new_population.append(child1)
            new_population.append(child2)

        population = sort_population(new_population)
        population = population[:populationsize]


        # search, if there is a new best individual
        for i in range(1, len(population)):
            if (population[i][6] < best_ind_fitness):
                # TODO check if I need deepcopy
                best_ind = population[i].copy()
                best_ind_fitness = best_ind[6]
                print('found new best individual')
                print('Fitness: ' + str(best_ind_fitness))

    # save result:
    params = ['U_before', 't_before', 'V_before', 'U_after', 't_after', 'V_after', 'Fitness']
    with open('parameter_genetic_tmp.txt', 'w') as f:
        for i in range(len(best_ind)):
            f.write(params[i] + ': ' + str(best_ind[i]) + '\n')

    return best_ind






def main():
    # n = number_sick_before[0] / float(population_germany)
    # const.n = n
    # number_trials = 10
    # result_before = list()
    # for trial in range(number_trials):
    #     best_ind = fitting_genetic(actual_number_sick=number_sick_before)
    #     result_before.append(best_ind)
    #     with open('parameter_genetic_before.txt', 'a') as f:
    #         f.write('Trial: ' + str(trial) + '\n')
    #         params = ['U', 't', 'V', 'Fitness']
    #         for i in range(len(best_ind)):
    #             f.write(params[i] + ': ' + str(best_ind[i]) + '\n')
    #
    # n = number_sick_after[0] / float(population_germany)
    # const.n = n
    # number_trials = 10
    # result_after = list()
    # for trial in range(number_trials):
    #     best_ind = fitting_genetic(actual_number_sick=number_sick_after)
    #     result_after.append(best_ind)
    #     with open('parameter_genetic_after.txt', 'a') as f:
    #         f.write('Trial: ' + str(trial) + '\n')
    #         params = ['U', 't', 'V', 'Fitness']
    #         for i in range(len(best_ind)):
    #             f.write(params[i] + ': ' + str(best_ind[i]) + '\n')

    number_trials = 5
    result = list()
    for trial in range(number_trials):
        best_ind = fitting_genetic_lockdown_percent(actual_number_sick=number_sick, percent=0.05)
        result.append(best_ind)
        with open('parameter_genetic_overall_factor' + str(const.factor_actual_cases) +'.txt', 'a') as f:
            f.write('Trial: ' + str(trial) + '\n')
            params = ['U_before', 't_before', 'V_before', 'U_after', 't_after', 'V_after', 'Fitness']
            for i in range(len(best_ind)):
                f.write(params[i] + ': ' + str(best_ind[i]) + '\n')

    print(result)
    #print(result_before)
    #print(result_after)

if __name__== "__main__":
    main()




















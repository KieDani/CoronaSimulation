import numpy as np
import main as ma
import constants as const
from multiprocessing import Pool
from scipy.optimize import curve_fit
from numba import jit

# Data from https://www.n-tv.de/infografik/Coronavirus-aktuelle-Zahlen-Daten-zur-Epidemie-in-Deutschland-Europa-und-der-Welt-article21604983.html
# from 08.03.2020 to 10.04.2020
number_sick_original = [1035, 1180, 1563, 1899, 2746, 3675, 4599, 5796, 7232, 9375, 12300, 15305, 19655, 22189, 21963, 26102, 30014, 34252, 37301, 42968, 48241, 52584, 48469, 50271, 53389, 57048, 60483, 63595, 65798, 66767, 67922, 59823, 60326, 61340]
number_immune_original = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2809, 2809, 2809, 2809, 5600, 5600, 5600, 5600, 13500, 16100, 18700, 21400, 23800, 26400, 28700, 30600, 33300, 46300, 50000, 53913]
population_germany = 80000000
faktor_actual_cases = 100


number_sick = np.asarray(number_sick_original) * faktor_actual_cases
number_immune = np.asarray(number_immune_original) * 10

number_sick = number_sick / float(population_germany) * const.numPop

print(number_sick)
print(len(number_sick))

n = number_sick[0] / float(population_germany)
print(n)


def simulate_multi(U, t, V):
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

    return arrayInfected

def simulate(U, t, V):
    number_processes = const.number_processes
    number_loops = number_processes
    numberDays = 34
    n = 3.234375e-07

    arrayInfected = ma.simulation(U=U, t=t, V=V, n=n, numberDays=numberDays, seed=42)
    arrayInfected = np.asarray(arrayInfected)
    for i in range(1, number_loops):
        arrayInfected += np.asarray(ma.simulation(U=U, t=t, V=V, n=n, numberDays=numberDays, seed=42))

    arrayInfected = arrayInfected / float(number_loops)

    return arrayInfected


def fitting_scipy(xdata, ydata):
    popt, pcov = curve_fit(simulate, xdata, ydata)
    print(popt)
    print(pcov)

xdata = range(len(number_sick))
#fitting_scipy(xdata=xdata, ydata=number_sick)















import numpy as np
import matplotlib.pyplot as plt
import Person as P
import constants as const
from multiprocessing import Pool
#import fitting
#from numba import jit


#ensure, that no position is taken twice in the beginning
def initializePosition(population, positionToID, graphSizeX, graphSizeY):
    while(True):
        pos = np.random.randint(graphSizeX, size=2)
        liste = positionToID[pos[1] * graphSizeX + pos[0]]
        if len(liste) == 0:
            return pos



def initializeInfection(numInfected, population):
    for i in range(numInfected):
        population[i].infect()
        xPos, yPos = population[i].getPos()
        #positionArray[xPos, yPos] *= -1



def indexToPosition(index, graphSizeX, graphSizeY):
    if(index == graphSizeX * graphSizeY):
        return graphSizeX, graphSizeY
    xPos = index % graphSizeX
    yPos = (index - xPos) / graphSizeX
    return xPos, yPos


def buildPositionArray(population, graphSizeX, graphSizeY):
    positionArray = np.zeros((graphSizeX, graphSizeY), dtype=np.int8)
    for ind in population:
        xPos, yPos = ind.getPos()
        # skip impurity
        if(xPos == graphSizeX and yPos == graphSizeY):
            continue
        if(ind.isInfected() == True):
            positionArray[xPos, yPos] -= 1
        else:
            positionArray[xPos, yPos] += 1
    return positionArray


#@jit(nopython=True)
def move(person, population, t, U, V, graphSizeX, graphSizeY, positionToID):
    r = np.random.rand()
    r2 = np.random.rand()
    # jump from impurity/supermarket
    if(person.atImpurity == True):
        if(r >= t and r < V+t):
            #oldX, oldY = person.getPos()
            xPos = np.random.randint(graphSizeX)
            yPos = np.random.randint(graphSizeY)
            #if new place is empty or people do not reject
            if(r2 > U or len(positionToID[yPos*graphSizeX + xPos])  == 0):
                positionToID[graphSizeY*graphSizeX].remove(person.getID())
                positionToID[yPos*graphSizeX + xPos].append(person.id)
                #infect other people
                if(person.isInfected() == True):
                    for member in positionToID[yPos*graphSizeX + xPos]:
                        population[member].infect()
                #get infected
                else:
                    for member in positionToID[yPos * graphSizeX + xPos]:
                        if(population[member].isInfected() == True):
                            person.infect()
                            break
                person.jumpFromImpurity(xPos=xPos, yPos=yPos)
    # jump to the impurity/supermarket
    elif(r >= t and r < V+t):
        oldX, oldY = person.getPos()
        if (r2 > U or len(positionToID[graphSizeY * graphSizeX]) == 0):
            positionToID[oldY * graphSizeX + oldX].remove(person.getID())
            positionToID[graphSizeY * graphSizeX].append(person.id)
            # infect other people
            if (person.isInfected() == True):
                for member in positionToID[graphSizeY * graphSizeX]:
                    population[member].infect()
            # get infected
            else:
                for member in positionToID[graphSizeY * graphSizeX]:
                    if (population[member].isInfected() == True):
                        person.infect()
                        break
            person.jumpToImpurity()

        pass
    # jump around on lattice
    else:
        teff = t/4.
        xPos, yPos = person.getPos()
        oldX = xPos
        oldY = yPos
        if(r < teff):
            xPos +=1
        elif(r < 2*teff):
            xPos -= 1
        elif(r < 3* teff):
            yPos +=1
        elif(r < 4*teff):
            yPos -= 1
        else:
            #no change
            return

        yPos = yPos % graphSizeY
        xPos = xPos % graphSizeX

        #statePositionArray = positionArray[xPos, yPos]
        positionList = positionToID[yPos*graphSizeX + xPos]
        #if lattice site is not empty
        if(len(positionList) != 0):
            #if jump is not rejected
            if(r2 > U):
                # infect other people
                if (person.isInfected() == True):
                    for member in positionToID[yPos * graphSizeX + xPos]:
                        population[member].infect()
                # get infected
                else:
                    for member in positionToID[yPos * graphSizeX + xPos]:
                        if (population[member].isInfected() == True):
                            person.infect()
                            break
            #if jump is rejected
            else:
                xPos = oldX
                yPos = oldY


        person.setPos(xPos=xPos, yPos=yPos)
        # if (statePositionArray < 0):
        #     positionArray[xPos, yPos] -= 1
        # else:
        #     if(person.isInfected() == True):
        #         positionArray[xPos, yPos] -= 1
        #     else:
        #         positionArray[xPos, yPos] += 1
        #
        # if (positionArray[oldX, oldY] < 0):
        #     positionArray[oldX, oldY] += 1
        # else:
        #     positionArray[oldX, oldY] -= 1

        id = person.getID()
        positionToID[oldY * graphSizeX + oldX].remove(id)
        positionToID[yPos * graphSizeX + xPos].append(id)


def calculateNumberInfected(population):
    numInfected = 0
    for ind in population:
        if(ind.isInfected() == True):
            numInfected += 1
    return numInfected

def calculateNumberImmune(population):
    numImmune = 0
    for ind in population:
        if(ind.isImmune() == True):
            numImmune += 1
    return numImmune

def timestep(population):
    for ind in population:
        ind.timeStep()


#n...initially infected persons per population
#@jit(nopython=True)
def simulation(U, t, V, n, numberDays, seed):
    np.random.seed(seed)
    #initial values
    graphSizeX = const.graphSizeX
    graphSizeY = const.graphSizeY
    population = list()
    numPop = const.numPop
    lengthOfDay = const.lengthOfDay
    numInfected = int(np.ceil(n * numPop))
    # 0...empty, 1...taken once, -1...infected
    # positionArray = np.zeros((graphSizeX, graphSizeY), dtype=np.int8)
    # If person is at (x,y), put it in list at position y*graphsizeX + x
    positionToID = list()

    # initialize positionToID with empty lists
    for i in range(graphSizeX * graphSizeY):
        positionToID.append(list())
    positionToID.append(list())


    #put persons into lattice
    for id in range(0, numPop):
        #positionArray = buildPositionArray(population=population, graphSizeX=graphSizeX, graphSizeY=graphSizeY)
        pos = initializePosition(population=population, positionToID=positionToID, graphSizeX=graphSizeX, graphSizeY=graphSizeY)
        # positionArray[pos[0], pos[1]] += 1
        population.append(P.Person(id=id, xPos=pos[0], yPos=pos[1], graphSizeX=graphSizeX, graphSizeY=graphSizeY))
        positionToID[pos[1] * graphSizeX + pos[0]].append(id)


    #initially infect people
    initializeInfection(numInfected=numInfected, population=population)


    #move the people around
    arrayInfected = list()
    arrayInfected.append(numInfected)
    arrayImmune = list()
    arrayImmune.append(0)
    #print(positionToID)
    #positionArray = buildPositionArray(population=population, graphSizeX=graphSizeX, graphSizeY=graphSizeY)
    #print(positionArray)
    for j in range(numberDays-1):
        #print(j)
        timestep(population=population)
        for i in range(lengthOfDay):
            for ind in population:
                move(person=ind, population=population, t=t[j], U=U[j], V=V[j], graphSizeX=graphSizeX, graphSizeY=graphSizeY,
                     positionToID=positionToID)
        numInfected = calculateNumberInfected(population=population)
        numImmune = calculateNumberImmune(population)
        arrayInfected.append(numInfected)
        arrayImmune.append(numImmune)
    #positionArray = buildPositionArray(population=population, graphSizeX=graphSizeX, graphSizeY=graphSizeY)
    #print(positionArray)
    #print(positionToID)


    #print('------------------------')
    #print(arrayInfected)

    return arrayInfected, arrayImmune





def main():
    numberDays = len(const.number_sick) + 165
    U = list()
    t = list()
    V = list()
    for i in range(0, const.time_lockdown):
        U.append(const.U_before)
        t.append(const.t_before)
        V.append(const.V_before)
    for i in range(const.time_lockdown, len(const.number_sick) + 10):
        U.append(const.U_after)
        t.append(const.t_after)
        V.append(const.V_after)
    for i in range(len(const.number_sick) + 10, numberDays):
        U.append(const.U_before)
        t.append(const.t_before)
        V.append(const.V_before)
    n = const.n
    number_processes = const.number_processes

    poolarray = []
    for i in range(number_processes):
        poolarray.append((U,t,V,n, numberDays, 42+i))
    with Pool(processes=number_processes) as pool:
        result = pool.starmap(simulation, poolarray)

    arrayInfected = np.asarray(result[0][0])
    arrayImmune = np.asarray(result[0][1])
    for i in range(1, number_processes):
        arrayInfected += np.asarray(result[i][0])
        arrayImmune += np.asarray(result[i][1])
        #print(result[i])
    arrayInfected = arrayInfected / float(number_processes)
    arrayImmune = arrayImmune / float(number_processes)
    #print(np.asarray(arrayInfected[:len(const.number_sick)]) - np.asarray(const.number_sick))


    plt.plot(np.asarray(arrayInfected)/const.numPop, color='red')
    plt.plot(np.asarray(arrayImmune)/const.numPop, color='green')

    plt.plot(const.number_sick/const.numPop, color = 'darkred')
    plt.plot(const.number_immune/const.numPop, color = 'darkgreen')

    plt.show()

    # print(arrayInfected)
    # for i in range(number_loops - 1):
    #     arrayInfected = arrayInfected + np.asarray(simulation(U=U, t=t, V=V, n=0.01))
    #     print(arrayInfected)
    # #arrayInfected = np.asarray(arrayInfected)
    # arrayInfected = arrayInfected / float(number_loops)
    # print(arrayInfected)
    # plt.plot(arrayInfected)
    # plt.show()



if __name__== "__main__":
  main()
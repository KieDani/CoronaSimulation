import numpy as np
import matplotlib.pyplot as plt
import Person as P
from numba import jit


#ensure, that no position is taken twice in the beginning
def initializePosition(population, positionArray, graphSizeX, graphSizeY):
    while(True):
        pos = np.random.randint(graphSizeX, size=2)
        if(positionArray[pos[0], pos[1]] == 0):
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

def timestep(population):
    for ind in population:
        ind.timeStep()


#n...initially infected persons per population
def simulation(U, t, V, n):
    #initial values
    np.random.seed(42)
    graphSizeX = 100
    graphSizeY = 100
    population = list()
    numPop = 2000
    lengthOfDay = 4
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
        positionArray = buildPositionArray(population=population, graphSizeX=graphSizeX, graphSizeY=graphSizeY)
        pos = initializePosition(population=population, positionArray=positionArray, graphSizeX=graphSizeX, graphSizeY=graphSizeY)
        # positionArray[pos[0], pos[1]] += 1
        population.append(P.Person(id=id, xPos=pos[0], yPos=pos[1], graphSizeX=graphSizeX, graphSizeY=graphSizeY))
        positionToID[pos[1] * graphSizeX + pos[0]].append(id)


    #initially infect people
    initializeInfection(numInfected=numInfected, population=population)


    #move the people around
    arrayInfected = list()
    arrayInfected.append(numInfected)
    #print(positionToID)
    #positionArray = buildPositionArray(population=population, graphSizeX=graphSizeX, graphSizeY=graphSizeY)
    #print(positionArray)
    for j in range(60):
        print(j)
        timestep(population=population)
        for i in range(lengthOfDay):
            for ind in population:
                move(person=ind, population=population, t=t, U=U, V=V, graphSizeX=graphSizeX, graphSizeY=graphSizeY,
                     positionToID=positionToID)
        numInfected = calculateNumberInfected(population=population)
        arrayInfected.append(numInfected)
    #positionArray = buildPositionArray(population=population, graphSizeX=graphSizeX, graphSizeY=graphSizeY)
    #print(positionArray)
    #print(positionToID)


    print('------------------------')
    print(arrayInfected)

    return arrayInfected





def main():
    U = 0.5
    t = 0.01
    V = 0.005
    np.random.seed(42)
    graphSizeX = 30
    graphSizeY = 30
    population = list()
    numPop = 500
    numInfected = 3
    arrayInfected = simulation(U=U, t=t, V=V, n=0.01)
    plt.plot(arrayInfected)
    plt.show()




main()
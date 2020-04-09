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



def indexToPosition(index, graphSizeX):
    xPos = index % graphSizeX
    yPos = (index - xPos) / graphSizeX
    return xPos, yPos


def buildPositionArray(population, graphSizeX, graphSizeY):
    positionArray = np.zeros((graphSizeX, graphSizeY), dtype=np.int8)
    for ind in population:
        xPos, yPos = ind.getPos()
        if(ind.isInfected() == True):
            positionArray[xPos, yPos] -= 1
        else:
            positionArray[xPos, yPos] += 1
    return positionArray


def move(person, population, positionArray, t, U, graphSizeX, graphSizeY, positionToID):
    teff = t/4.
    xPos, yPos = person.getPos()
    oldX = xPos
    oldY = yPos
    r = np.random.rand()
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

    statePositionArray = positionArray[xPos, yPos]
    if(statePositionArray != 0):
        r2 = np.random.rand()
        if(r2 > U):
            if(statePositionArray < 0):
                person.infect()
            else:
                if(person.isInfected() == True):
                    for member in positionToID[yPos*graphSizeX + xPos]:
                        population[member].infect()
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

    #person.setPos(xPos=xPos, yPos=yPos)

def calculateNumberInfected(population):
    numInfected = 0
    for ind in population:
        if(ind.isInfected() == True):
            numInfected += 1
    return numInfected

def timestep(population):
    for ind in population:
        ind.timeStep()


def main():
    #initial values
    U = 0.2
    t = 0.6
    np.random.seed(42)
    graphSizeX = 30
    graphSizeY = 30
    population = list()
    numPop = 500
    numInfected = 3
    # 0...empty, 1...taken once, -1...infected
    # positionArray = np.zeros((graphSizeX, graphSizeY), dtype=np.int8)
    # If person is at (x,y), put it in list at position y*graphsizeX + x
    positionToID = list()

    # initialize positionToID with empty lists
    for i in range(graphSizeX * graphSizeY):
        positionToID.append(list())


    #put persons into lattice
    for id in range(0, numPop):
        positionArray = buildPositionArray(population=population, graphSizeX=graphSizeX, graphSizeY=graphSizeY)
        pos = initializePosition(population=population, positionArray=positionArray, graphSizeX=graphSizeX, graphSizeY=graphSizeY)
        # positionArray[pos[0], pos[1]] += 1
        population.append(P.Person(id=id, xPos=pos[0], yPos=pos[1]))
        positionToID[pos[1] * graphSizeX + pos[0]].append(id)


    #initially infect people
    initializeInfection(numInfected=numInfected, population=population)


    #move the people around
    arrayInfected = list()
    arrayInfected.append(numInfected)
    print(positionToID)
    positionArray = buildPositionArray(population=population, graphSizeX=graphSizeX, graphSizeY=graphSizeY)
    print(positionArray)
    for j in range(100):
        print(j)
        timestep(population=population)
        for ind in population:
            # this makes everything quadratic :(
            positionArray = buildPositionArray(population=population, graphSizeX=graphSizeX, graphSizeY=graphSizeY)
            move(person=ind, population=population, positionArray=positionArray, t=t, U=U, graphSizeX=graphSizeX, graphSizeY=graphSizeY,
                 positionToID=positionToID)
        numInfected = calculateNumberInfected(population=population)
        arrayInfected.append(numInfected)
    positionArray = buildPositionArray(population=population, graphSizeX=graphSizeX, graphSizeY=graphSizeY)
    print(positionArray)
    print(positionToID)


    print('------------------------')
    print(arrayInfected)




main()
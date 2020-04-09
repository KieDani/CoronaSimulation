from numba import jit

class Person:

    def __init__(self, id, xPos, yPos, graphSizeX, graphSizeY):
        self.id = id
        self.xPos = xPos
        self.yPos = yPos
        self.infected = False
        self.timeInfected = 0
        self.immune = False
        self.durationSickness = 20
        self.atImpurity = False
        self.graphSizeX = graphSizeX
        self.graphSizeY = graphSizeY


    def getPos(self):
        return self.xPos, self.yPos

    def setPos(self, xPos, yPos):
        self.xPos = xPos
        self.yPos = yPos

    def getID(self):
        return self.id

    def isImmune(self):
        return self.immune

    def isInfected(self):
        return self.infected

    def infect(self):
        if(self.immune == False):
            self.infected = True

    def timeStep(self):
        if(self.infected == True):
            self.timeInfected += 1

        if(self.timeInfected > 20):
            self.infected = False
            self.immune = True

    def jumpToImpurity(self):
        self.atImpurity = True
        self.xPos = self.graphSizeX
        self.yPos = self.graphSizeY

    def jumpFromImpurity(self, xPos, yPos):
        self.xPos = xPos
        self.yPos = yPos
        self.atImpurity = False

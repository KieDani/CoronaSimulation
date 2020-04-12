import constants as const
#from numba import jit, jitclass
#from numba import int32, float32, boolean

class Person:

    # spec = [
    #     ('id', int32),
    #     ('xPos', int32),
    #     ('yPos', int32),
    #     ('infected', boolean),
    #     ('timeInfected', int32),
    #     ('immune', boolean),
    #     ('durationSickness', int32),
    #     ('atImpurity', boolean),
    #     ('graphSizeX', int32),
    #     ('graphSizeY', int32),
    # ]

    #@jitclass(spec)
    def __init__(self, id, xPos, yPos, graphSizeX, graphSizeY):
        self.id = id
        self.xPos = xPos
        self.yPos = yPos
        self.infected = False
        self.timeInfected = 0
        self.immune = False
        self.durationSickness = const.durationSickness
        self.atImpurity = False
        self.graphSizeX = graphSizeX
        self.graphSizeY = graphSizeY

    #@property
    def getPos(self):
        return self.xPos, self.yPos


    def setPos(self, xPos, yPos):
        self.xPos = xPos
        self.yPos = yPos

    #@property
    def getID(self):
        return self.id

    #@property
    def isImmune(self):
        return self.immune

    #@property
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

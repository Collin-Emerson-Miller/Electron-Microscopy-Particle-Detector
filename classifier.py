import numpy as np
import random;

class classifier:

    def __int__(self, image):
        self.image = image


    def c(self):
        percent = 10
        prob = random.randrange(0,100)
        if prob > percent:
            return True
        else:
            return False


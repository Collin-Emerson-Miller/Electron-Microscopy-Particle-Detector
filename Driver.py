from classifier import classifier
from SlidingWindow import SlidingWindow
import numpy as np
import sys
import pprint


class Driver():




    def main(self):

        image = np.random.rand(256, 256)
        # np.set_printoptions(precision=3, threshold=256, edgeitems=128, linewidth=256)
        # pprint.pprint(image)

        blc = 32
        c = classifier
        shift_r = 3
        shift_d = 3

        s_w = SlidingWindow()
        s_w.blc=blc
        s_w.image=image
        s_w.classifier=c
        s_w.shift_r=shift_r
        s_w.shift_d=shift_d

        s_w.move_slider()




if __name__ == '__main__':
    Driver().main()



















import numpy as np
import sys


class SlidingWindow(object):

    def __int__(self, blc, image, classifier, shift_r, shift_d):

        self.blc = blc
        self.image = image
        self.classifier = classifier
        self.shift_r = shift_r
        self.shift_d = shift_d

        """"
        BLC : Bottom Left Corner : This indicates the start point of the sliding window and takes in a value such as 32
        Image : This is the image that will be passed as a numpy array
        Classifier : This is the reference from the CNN handling the identification
        ShiftR : Shift Right : This is how many pixels the image will shift to the right
        ShiftD : Shift Down : This is how many pixels the image will shift down
        """


    def move_slider(self):

        count_x = 0  # number to increase each iteration of while
        count_y = 0
        shape_of_array = np.shape(self.image)  # the dimensions of the numpy Image array
        max_x = shape_of_array[0]  # the dimension along the x axis
        max_y = shape_of_array[1]  # the dimension along the y axis

        slice_obj = (slice(0, self.blc-1, 1), slice(0, self.blc-1, 1))  # sets the slice of 0 to the BLC of each axis
        sub_image = self.image[slice_obj]  # gets the pixel values from the 0 index to the max index of the passed BLC

        blc_x = self.blc
        blc_y = self.blc


        while blc_y < max_y:

            # first run through starts at 0,0 and has a BLC of what was passed in i.e. 32x32
            # once that has run through this while loop lowers the sub_image(0, ShiftD)

            new_y_range = (slice(count_y * self.shift_d, self.blc - 1 + (count_y * self.shift_d), 1))
            # new_y_range is the slice of the y axis from the start of the count to the BLC provided


            while blc_x < max_x:

                # as stated the first run through the top left sets to 0,0
                # in this case the window will slide to the right a number of pixels equal to shift_r

                # first iteration: (0, 0) through (31, 31)
                # second iteration: (shiftR, 0) through (31+ShiftR, 32)

                new_x_range = (slice(count_x * self.shift_r, self.blc - 1 + (count_x * self.shift_r), 1))
                # new_x_dim is the slice of the y axis from the start of the count to the BLC provided

                slice_obj = (new_x_range, new_y_range)

                sub_image = self.image[slice_obj]

                c = self.classifier()
                c.image=sub_image

                output_string = "{}\n".format(c.c())
                sys.stdout.write(output_string)
                sys.stdout.flush()

                count_x += 1
                blc_x = self.blc + (count_x * self.shift_r)

            count_y += 1
            if blc_y < max_y:
                blc_y = self.blc + (count_y * self.shift_d)

            if blc_x >= max_x:
                blc_x = self.blc
                count_x = 0
import numpy as np
import csv
import matplotlib.pyplot as plt

with open('train.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    # for i in range(7):
    #     next(reader)
    for data in reader:
        # The second column is the label
        label = data[1]

        # The rest of columns are pixels
        pixels = data[2:]

        # Make those columns into a array of 8-bits pixels
        # This array will be of 1D with length 784
        # The pixel intensity values are integers from 0 to 255
        pixels = np.array(pixels, dtype='uint8')
        pixels = pixels.reshape((28, 28))

        # Plot
        plt.title('Label is {label}'.format(label=label))
        plt.imshow(pixels, cmap='gray')
        plt.show()

        break 
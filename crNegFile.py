import os
import numpy as np


for fileType in ['positive_images']:

    for img in os.listdir(fileType):

        line = fileType + '/' + img + '\n'

        with open('positives.txt', 'a') as f:
            f.write(line)
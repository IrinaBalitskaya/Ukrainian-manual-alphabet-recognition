import pandas as pd
import os
import re


def datasetCreation():
    directory = './images/'

    imagesFolders = os.listdir(directory)

    for fldr in imagesFolders:
        csv_data = {'filename': [], 'class': []}
        fileNames = [os.path.basename(filename) for filename in os.listdir(directory + fldr)]
        csv_data['filename'].extend(fileNames)
        for flnm in fileNames:
            className = int(re.findall(r'_(.*?)_', flnm)[0]) - 1
            csv_data['class'].append(className)

        df = pd.DataFrame(data=csv_data)
        df.to_csv(directory + fldr + '_labels.csv', index=False)


if __name__ == '__main__':
    datasetCreation()
    print("Datasets were created")

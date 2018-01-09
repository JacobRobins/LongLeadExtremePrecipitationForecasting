#This script is for creating the class labels
import csv
import itertools
#We just need the iowa data for 31 years for creating the class labels
fileName = "raw_data/iowa_31years.csv"
sumList = []
for time in range(0,11309):
    with open(fileName, 'r') as csvFileIn:
        csvReader = csv.reader(csvFileIn)
        sum = 0
        #take 15 days sum and put into sum
        for row in itertools.islice(csvReader, time, time+15):
            sum = sum + float(row[0]) 
    #append each sum to sumList
    sumList.append(sum)

import numpy as np
#convert sumList to numpy array, so as to do percentile
sumArray = np.array(sumList)
#p is the 95th percentile
p = np.percentile(sumArray, 95)

classLabel = []
for row in sumList:
    #if sum is > than p, classLabel is 1. Else its 0
    if row > p:
        classLabel.append(1)
    else:
        classLabel.append(0)

#we need only class labels from 1-15-1980, so remove first 14 rows
classLabel = classLabel[14:]
sumList = sumList[14:]

#Write into class_label.csv
with open("classLabels.csv", 'wb') as csvFileOut:
    csvWriter = csv.writer(csvFileOut)
    for label in classLabel:
        csvWriter.writerow([label])
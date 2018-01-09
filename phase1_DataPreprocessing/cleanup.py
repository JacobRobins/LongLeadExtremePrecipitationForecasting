#This script clean up 31years data for each 9 meteo variables by removing blank rows and the first 4 columns
import csv
fileNames = ["z300", "z500", "z1000", "u300", "v300", "u850", "v850", "t850" ,"pw", "iowa"]

for fileName in fileNames:
    fileName1 = "raw_data/" + fileName + "_31years.csv"
    with open(fileName1, 'rb') as csvFileIn:
        csvReader = csv.reader(csvFileIn)
        for row in csvReader:
            #consider only non-blank rows
            if row[0] not in '':
                #delete first 4 columns
                del row[:4]
                with open(fileName + "_31years.csv", 'ab') as output:
                    csvWriter = csv.writer(output)
                    csvWriter.writerow(row)
#This script is used to create the headers for the feature table
#The headers will be in the format T9z3S1 T9z3S2 T9z3s3 .....
import csv
meteoVariable = ["z3", "z5", "z1", "u3", "v3", "u8", "v8", "t8", "pw"]
time = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
header = []
#We need the header so as to keep track of the columns in the feature table(for SAOLA)
for t in time:
    s = "T" + str(t)
    for mv in meteoVariable:
        s = s + mv + "S"
        for location in range(1,5329):
            temp = s
            temp = temp + str(location)
            header.append(temp)
        s = "T" + str(t)

with open("header.csv", 'wb') as csvFileOut:
    csvWriter = csv.writer(csvFileOut)
    csvWriter.writerow(header)
#This script is used to create the t9 table. This table will contain 5328*9 columns and 11323 rows
import pandas as pd
fileNames = ["z1000", "u300", "v300", "u850", "v850", "t850" ,"pw"]

df = pd.read_csv("raw_data/z300_31years.csv", header = None)
df1 = pd.read_csv("raw_data/z500_31years.csv", header = None)
#we initially create result with concat of z300_31years and z500_31years
result = pd.concat([df, df1], axis = 1)

#For rest of 7 meteo variable we doe concat in the following loop
for fileName in fileNames:
    df = pd.read_csv("raw_data/" + fileName + "_31years.csv", header = None)
    result = pd.concat([result, df], axis = 1)

#Write result to get the t9.csv
result.to_csv('t9.csv', index = False)
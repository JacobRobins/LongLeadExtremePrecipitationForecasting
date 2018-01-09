#This script creates the feature table using t9 table(11323 rows and 5328*9 columns)
import pandas as pd
import numpy

#idea is that each row in the final table is constructed from only 10 rows, so extract only 10.
for m in range(0,11295):
    df = pd.read_csv('t9.csv', skiprows = m, nrows=10)
    #this is done to create a dummy dataframe with only a single element(0).
    final = numpy.zeros(shape=(1,1))
    df_final = pd.DataFrame(final)
    for time in range(0,10):
        #We cut out the timeth row
        temp= df.iloc[time:time+1,:]
        #We need to concatenate timeth row at the end of dummy table, so index is made to 0
        temp.index = [0]
        df_final = pd.concat([df_final,temp], axis = 1)
    #this step is done to remove the 0 we initially created     
    del df_final[0]
    #At this moment df_final will have one complete row of our final_table, so append it to output.csv file.
    df_final.to_csv('features.csv', mode='a', header=False, index=False)


    

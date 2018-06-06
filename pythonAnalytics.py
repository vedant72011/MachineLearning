import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use("fivethirtyeight")

df = pd.DataFrame({"Day":[1,2,3,4], "Visitors":[200,300,100,205], "BounceRate":[15,20,30,15]})

df.set_index("Day", inplace=True)

df.rename(columns={"Visitors":"Users"})

print(df)


df1 = pd.DataFrame({'HPI':[80,85,88,85], 'Int_rate':[2,3,2,2], 'US_GDP_Thousands':[50, 55, 65, 55]}
                   ,index=[2001, 2002, 2003, 2004])

df2 = pd.DataFrame({'HPI':[80,85,88,85], 'Int_rate':[2,3,2,2], 'US_GDP_Thousands':[50, 55, 65, 55]},
                   index=[2005, 2006, 2007, 2008])

concat = pd.concat([df1, df2])

print(concat)

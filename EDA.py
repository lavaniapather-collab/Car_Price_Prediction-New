import os
print("Current working directory:", os.getcwd())
import pandas as pd
df=pd.read_csv("car_dataset.csv")
df.head()
df['Selling_Price']=df['Selling_Price']*100000
df['Present_Price']=df['Present_Price']*100000

from datetime import date
  
# creating the date object of today's date
todays_date = date.today()
print(todays_date)

df["Car_Age"]=todays_date.year-df['Year']
df.head()


df=pd.get_dummies(df.drop(columns=['Car_Name','Seller_Type'],axis=1), dtype=int)
#this function converts categorical variables into dummy (indicator) variables — also known as one-hot
df.head()
df.to_csv("car_cleaned_data.csv")
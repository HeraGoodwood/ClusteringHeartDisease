import pandas as pd

df = pd.read_csv(r"C:\Users\User\Desktop\Capstone Project\health.csv") 
print(df.head(10))

#remove unwanted columns 
to_drop = ['Class', 'Class Language', 'Year', 'Insurance Category', 'Medical Home Category', 
           'Race/Ethnicity', 'Education Level','Previous Diabetes Education (Yes/No)', 
           'Diabetes Knowledge','Food Measurement', 
           'Carbohydrate Counting','Problem Area in Diabetes (PAID) Scale Score', 
           'ZIP code (address)', 'ZIP code (city)','ZIP code (state)', 'ZIP code (zip)']

df.drop(to_drop, inplace=True, axis=1)
print(df.head(10))

#check number of missing values in each column 
print(df.isnull().sum())

#drop any row that has NaN values
df = df.dropna()
print(df)


#standardize values 
df['Exercise'] = df['Exercise'].replace( # Changing the format of the string
                                      to_replace=['0 days','1 day', '2 days',
                                                  '3 days', '4 days', '5 or more days',], 
                                      value=['0','1', '2', '3', '4', '>=5'])

df['Sugar-Sweetened Beverage Consumption'] = df['Sugar-Sweetened Beverage Consumption'].replace(to_replace=['3 or more'], 
                                             value=['>=3'])

print(df)

#save the cleaned dataset
df.to_csv(r"C:\Users\User\Desktop\Capstone Project\modified_health.csv")

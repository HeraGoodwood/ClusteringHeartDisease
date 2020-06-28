import pandas as pd  
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

def count(predicts):
	c = 0
	for pre in predicts:
		if pre == True:
			c+=1
	return c

#importing dataset and converting to dataframe
df = pd.read_csv(r"C:\Users\User\Documents\BSc(AI)\SEM 4\Introduction to Data Science\Capstone Project\modified_health.csv")
df.dtypes
number= LabelEncoder()
df['Gender']=number.fit_transform(df['Gender'].astype('str'))
df['Diabetes Status (Yes/No)']=number.fit_transform(df['Diabetes Status (Yes/No)'].astype('str'))
df['High Blood Pressure (Yes/No)']=number.fit_transform(df['High Blood Pressure (Yes/No)'].astype('str'))
df['Heart Disease (Yes/No)']=number.fit_transform(df['Heart Disease (Yes/No)'].astype('str'))
df['Tobacco Use (Yes/No)']=number.fit_transform(df['Tobacco Use (Yes/No)'].astype('str'))
df['Fruits & Vegetable Consumption']=number.fit_transform(df['Fruits & Vegetable Consumption'].astype('str'))
df['Exercise']=number.fit_transform(df['Exercise'].astype('str'))
data = pd.DataFrame(df) #data frame

#extracting columns x and y
x = data.iloc[:, 0:8]
x = x.drop(columns=['Heart Disease (Yes/No)'], axis=1)
print(x.dtypes)
print ("======================================================================")
x = pd.DataFrame(scale(x))


y = data.iloc[:, 3]
print(y.head())
print ("======================================================================")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=50)

# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
# predictions
rfc_predict = rfc.predict(x_test)

rfc_cv_score = cross_val_score(rfc, x, y, cv=8, scoring='roc_auc')

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

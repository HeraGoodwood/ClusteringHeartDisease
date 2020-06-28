import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize,scale
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import timeit

start = timeit.default_timer()
def count(predicts):
	c = 0
	for pre in predicts:
		if pre == True:
			c+=1
	return c

#importing dataset and converting to datasframe
df = pd.read_csv('modified_health.csv')
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


#plotting the data
fig = plt.figure()


clusters = 2

model = KMeans(init='k-means++', n_clusters=clusters,
               random_state=20, max_iter=70)

scores = cross_val_score(model, x, y, scoring='accuracy', cv=8)
print ("8-Fold Accuracy : ", scores.mean()*100)
print ("======================================================================")
model.fit(x)

predicts = model.predict(x)
print ("Accuracy(Total) = ", count(predicts == np.array(y))/(len(y)*1.0) *100)
centroids = model.cluster_centers_
print ("======================================================================")
# print centroids
print(x[1].head)

ax2 = fig.add_subplot(1,1,1)
ax2.set_title("KMeans Clustering")
ax2.scatter(x[1],x[2], c=predicts)

ax2.scatter(x[1],x[3], c=predicts)


ax2.scatter(x[1],x[4], c=predicts)

ax2.scatter(x[1],x[5], c=predicts)

ax2.scatter(x[1],x[6], c=predicts)

ax2.scatter(centroids[:, 1], centroids[:, 2],
            marker='x', s=169, linewidths=3,
            color='b', zorder=15)

#metrics
cm = metrics.confusion_matrix(y, predicts)
print (cm/len(y))
print ("======================================================================")
print (metrics.classification_report(y, predicts))
print ("======================================================================")
plt.show()
print ("======================================================================")
stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in "+str(execution_time))
print ("======================================================================")

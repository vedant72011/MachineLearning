import scipy
import pandas
from pandas.tools.plotting import scatter_matrix
from matplotlib import pyplot as plt
from matplotlib import style as style
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

style.use("ggplot")
url = open("Traffic_Violations.csv")

names = ['Date Of Stop', 'Time Of Stop', 'Agency', 'SubAgency', 'Description', 'Location', 'Latitude', 'Longitude', 'Accident', 'Belts', 'Personal Injury',	'Property Damage', 'Fatal', 'Commercial License', 'HAZMAT', 'Commercial Vehicle', 'Alcohol', 'Work Zone', 'State', 'VehicleType', 'Year', 'Make', 'Model', 'Color', 'Violation Type', 'Charge',	'Article', 'Contributed To Accident', 'Race', 'Gender',	'Driver City', 'Driver State', 'DL State', 'Arrest Type', 'Geolocation']

dataSet = pandas.read_csv(url, names = names)

#dimensions
'print(dataSet.shape)'
#print a certain number of items from the data
'print(dataSet.head(10))'
#shows the count, the mean, min and max and other
'print(dataSet.describe())'
#grouping the dataset
print(dataSet.groupby('SubAgency').size())
print('------------------------------------')
#print(dataSet.groupby('HAZMAT').size())
print('------------------------------------')
#print(dataSet.groupby('Belts').size())
print('------------------------------------')
#print(dataSet.groupby('Personal Injury').size())
print('------------------------------------')
#print(dataSet.groupby('Property Damage').size())
print('------------------------------------')
#print(dataSet.groupby('Personal Injury').size())
print('------------------------------------')
#print(dataSet.groupby(['Work Zone', 'Location']).size())
print('------------------------------------')
#print(dataSet.groupby('Alcohol').size())
#print(dataSet.groupby('Date Of Stop').size())

x = [1,2,3,4,5,6,7]
y = [132311, 165278, 231184, 277597, 127250, 149399, 38439]

plt.bar(x,y, color='b')
plt.xlabel("SubAgency")
plt.ylabel("Number of Violations")
plt.legend()
plt.show()

#plotting dataset
plt.plot([dataSet.groupby('SubAgency')])
plt.show()


# Split-out validation dataset
"""array = dataSet.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring = 'accuracy'

print(X_validation)
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

test_prediction = knn.predict([[7.2, 3.0, 5.8, 1.5]])
print(test_prediction)"""

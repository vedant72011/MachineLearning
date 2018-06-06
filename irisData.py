import scipy
import pandas
import tensorflow as tf
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
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


url = open("irisFlower.csv")

names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

dataSet = pandas.read_csv(url, names = names)

#dimensions
'print(dataSet.shape)'
#print a certain number of items from the data
'print(dataSet.head(10))'
#shows the count, the mean, min and max and other
'print(dataSet.describe())'
#grouping the dataset
print(dataSet.groupby('class').size())
#plotting dataset
'dataSet.plot(kind = ''box'', subplots=True, layout=(2,2), sharex=False, sharey=False)'
'plt.show()'


# Split-out validation dataset
array = dataSet.values
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
print(test_prediction)

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2)

print(result)
sess = tf.Session()
print(sess.run(result))
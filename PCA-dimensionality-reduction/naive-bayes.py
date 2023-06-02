import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import os
a= "dataset-cropped-v3.csv"
dataframe = pandas.read_csv(a)
dataset = dataframe.values
X = dataframe.loc[:, dataframe.columns != 'class']
y = dataframe.loc[:, dataframe.columns == 'class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 1)
classes = ["B1972VBA", "BG1468AX", "L1412EY", "L1559AAT", "W1025ED", "W1046BM"]
n_classes = len(classes)

# create model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
clf = model.fit(X_train, y_train)

#------------
y_pred = model.predict(X_test)

# Use score method to get accuracy of the model
cm = confusion_matrix(y_test, y_pred)
print (cm)

# Use accuracy_score to get accuracy of the model
acc = accuracy_score(y_test, y_pred)
print('Accuracy Score: ', acc)
print(classification_report(y_test, y_pred))
y_score = model.predict_proba(X_test)
from sklearn.preprocessing import label_binarize

# Binarize the output
y_test = label_binarize(y_test, classes=classes)
from matplotlib import pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
plt.style.use('ggplot')

# Plotting and estimation of FPR, TPR
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
  fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
  roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green', 'yellow'])
for i, color in zip(range(n_classes), colors):
  plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(classes[i], roc_auc[i]))
  
plt.plot([0, 1], [0, 1], 'k-', lw=1.5)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()
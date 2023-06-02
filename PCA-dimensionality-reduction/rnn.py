# multi-class classification with Keras
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

accuracy = 0
#while accuracy < 0.90:
  dataframe = pandas.read_csv("datasetnew1.csv")
  dataset = dataframe.values
  column = 7120
  classlabel = 6
  # X = dataset[:,0:ilabel].astype(float)
  # Y = dataset[:,ilabel]
  x=dataframe.copy()
  y=x.pop('class')
  classes = ["B1972VBA", "BG1468AX", "L1412EY", "L1559AAT", "W1025ED", "W1046BM"]
  # Binarize the output
  y_bin = label_binarize(y, classes=classes)
  n_classes = y_bin.shape[1]
  # encode class values as integers
  encoder = LabelEncoder()
  encoder.fit(y)
  encoded_Y = encoder.transform(y)

  # convert integers to dummy variables (i.e. one hot encoded)
  dummy_y = np_utils.to_categorical(encoded_Y)

  # create model
  model = Sequential()
  model.add(Dense(128, input_dim=column, activation='relu'))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(16, activation='relu'))

  # model.add(Dense(8, input_dim=ilabel, activation='relu'))
  model.add(Dense(classlabel, activation='softmax'))

  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  nepochs = 200
  nbatch = 5

  # ------------ menggunakan packages
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.2)

  model.fit(X_train, y_train, epochs=nepochs, batch_size=nbatch)
  _, accuracy = model.evaluate(X_test, y_test)
  print('Accuracy: %.2f' % (accuracy*100))

  y_score = model.predict(X_test)
  y_score
  _, accuracy = model.evaluate(X_test, y_test)



model.save("my_model")
model.save_weights("model.h5")
print('Accuracy: %.2f' % (accuracy*100) + '%')

#plotting
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(classlabel):
  fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
  roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green', 'yellow', 'cyan', 'purple'])
for i, color in zip(range(n_classes), colors):
  plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label='ROC curve of class {0} (AUC = {1:0.2f})' ''.format(classes[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k-', lw=1.5)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()
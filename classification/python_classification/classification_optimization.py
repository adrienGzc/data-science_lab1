import pathlib
import pandas
import numpy
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

FILE_NAME = './titanic3.csv'
RANDOM_SEED = 1486310506

def getDataFromCSV(filename, featureToDrop=[]):
  if filename == '':
    return None
  data = pandas.read_csv(filename)

  # Drop unused column
  if featureToDrop != []:
    for feature in featureToDrop:
      del data[feature]
  return data

def convertToNumeric(data):
  sexMapping = { 'male': 0, 'female': 1 }
  embarkedMapping = { 'S': 1, 'C': 2, 'Q': 3 }

  # Fill NaN value with the average value of different field. Harry Potter here...
  data['embarked'] = data['embarked'].fillna('S')
  data['age'].fillna(data.groupby('title')['age'].transform('median'), inplace=True)
  data['fare'].fillna(data.groupby('pclass')['fare'].transform('median'), inplace=True)

  # Replace strings with number to get a full number dataset. No magic, just mapping...
  data['sex'] = data['sex'].map(sexMapping)
  data['embarked'] = data['embarked'].map(embarkedMapping)

def convertStringToFloat(data):
  tmp = pandas.DataFrame(data)
  for feature in tmp:
    tmp[feature] = tmp[feature].astype(float)

def replaceTitleNameByNumeric(data):
  # extract title from name and create new column with
  data['title'] = data['name'].str.extract(r'([A-Za-z]+)\.', expand=False)

  # Convert title into numeric value and delete name column
  # 0 = men, 1 = ladys, 2 = woman(older), 3 = others
  # Look at the output: data['title'].value_counts()
  titleMapping = {
    'Mr': 0,
    'Miss': 1,
    'Mrs': 2,
    'Master': 3,
    'Dr': 3,
    'Rev': 3,
    'Col': 3,
    'Major': 3,
    'Mlle': 3,
    'Countess': 3,
    'Ms': 3,
    'Lady': 3,
    'Jonkheer': 3,
    'Don': 3,
    'Dona' : 3,
    'Mme': 3,
    'Capt': 3,
    'Sir': 3
  }
  data['title'] = data['title'].map(titleMapping)
  del data['name']

def purgeData(data):
  replaceTitleNameByNumeric(data)
  convertToNumeric(data)
  convertStringToFloat(data)

def getClassifierPrediction(data, targetData, classifiers=[]):
  tmp = []
  # Param for cross-validation-scorer
  kFold = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

  for clf in classifiers:
    tmp.append(cross_val_predict(clf, data, targetData, cv=kFold, n_jobs=4))
  return tmp

def displayROCCurve(scoreList, targetData, options=[]):
  # Init ROC figure
  plt.figure()
  plt.title('ROC curve')
  plt.plot([0, 1], [0, 1], 'k--')
  plt.ylabel('True positive rate')
  plt.xlabel('False positive rate')

  for index in range(len(scoreList)):
    fpr, tpr, _ = roc_curve(targetData, scoreList[index])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=options[index]['label'] + ': %0.2f' % roc_auc, color=options[index]['color'])

  plt.legend(loc="lower right")
  plt.show()


if __name__ == '__main__':
  print('Welcome to DisneyLand!\nGetting data...')
  data = getDataFromCSV(FILE_NAME, ['boat', 'home.dest', 'body', 'ticket', 'cabin'])
  print('Done!')
  print('Purging the data...')
  purgeData(data)
  print('Finish purging the data.\n')

  # extract column survived for target column prediction and delete it from the dataset
  targetData = data['survived']
  del data['survived']

  # n_estimators=100, max_depth=6, random_state=RANDOM_SEED
  classifierList = [
    RandomForestClassifier(
      n_estimators=100,
      criterion='entropy',
      max_depth=10,
      n_jobs=4,
      random_state=RANDOM_SEED,
    ),
  ]
  allScore = getClassifierPrediction(data, targetData, classifierList)

  # Display ROC Curve
  optionsROCCurve = []
  for index in range(len(allScore)):
    optionsROCCurve.append({'label': 'Random Forest', 'color': 'r'})
  displayROCCurve(allScore, targetData, optionsROCCurve)
import pathlib
import pandas
import numpy as np
from scipy import interp
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
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

def getAUCClassifier(data, targetData, classifiers=[], display=True):
  # Init ROC figure
  plt.figure()
  plt.title('ROC curve')
  plt.plot([0, 1], [0, 1], 'k--')
  plt.ylabel('True positive rate')
  plt.xlabel('False positive rate')
  cv = StratifiedKFold(n_splits=30)

  tprs = []
  aucs = []
  for clf in classifiers:
    mean_fpr = np.linspace(0, 1, 20)
    for train, test in cv.split(data, targetData):
      data_train, data_test = data.iloc[train], data.iloc[test]
      targetData_train, targetData_test = targetData[train], targetData[test]
      probas = clf['instance'].fit(data_train, targetData_train).predict_proba(data_test)
      fpr, tpr, _ = roc_curve(targetData_test, probas[:, 1])
      tprs.append(interp(mean_fpr, fpr, tpr))
      tprs[-1][0] = 0.0
      roc_auc = auc(fpr, tpr)
      aucs.append(roc_auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, lw=3, alpha=0.7, label=clf['label'] + ': %0.2f' % mean_auc, color=clf['color'])

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

  classifierList = [
    {
      'label': 'Decision Tree',
      'instance': DecisionTreeClassifier(random_state=RANDOM_SEED),
      'color': 'r'
    },
    {
      'label': 'Random Forest',
      'instance': RandomForestClassifier(
        n_estimators=100,
        criterion='entropy',
        max_depth=10,
        n_jobs=4,
        random_state=RANDOM_SEED),
      'color': 'b'
    },
    {
      'label': 'Naive Bayes',
      'instance': GaussianNB(priors=None),
      'color': 'g'
    }
  ]
  getAUCClassifier(data, targetData, classifierList)
  # displayROCCurve(allScore, targetData, optionsROCCurve)

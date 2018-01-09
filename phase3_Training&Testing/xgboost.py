#XgBoost classifier
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.combine import SMOTETomek
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

#plotting the confusion matrix
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cnf_matrix.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#setting seed for reproducing
seed =123
#loading the datasets
target = pd.DataFrame(np.load('target_1980_2010.npy').astype(float))
data = pd.DataFrame(np.load('data_selected_1980_2010.npy').astype(float))
target = target.iloc[:, 1:]
#splitting the datasets in train and test sets
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=1803, random_state=seed)
#generate minority class and undersample majority to balance the classes
sm = SMOTETomek()
data_train_s, target_train_s = sm.fit_sample(data_train, target_train)
data_train_s = pd.DataFrame(data_train_s)
target_train_s = pd.DataFrame(target_train_s)
#creating xgboost matrices
data_test = pd.DataFrame(data_test)
dtrain = xgb.DMatrix(data_train_s, label=target_train_s)
dtest = xgb.DMatrix(data_test)
type (dtrain)
train_labels = dtrain.get_label()
#parameters for xgboost
params = {
        'objective':'binary:logistic',
        'max_depth':5,
        'silent': 1,
        'n_estimators' : 1000,
        'learning_rate': 0.1,
        'min_child_weight': 1,
        'gamma' : 0,
        'subsample' : 0.8,
        'colsample_bytree':0.8,
        #'eta':1,
        'nthread':4,
       # 'max_delta_step' :1,
      'eval_metric':'auc'   
      }
num_rounds = 20
#training and predicting the model
bst = xgb.train(params, dtrain, num_rounds)
y_test_preds = (bst.predict(dtest) > 0.49).astype(int)
cnf_matrix = confusion_matrix(target_test, y_test_preds)
plot_confusion_matrix(cnf_matrix, ['0', '1'])

# Compute ROC curve and area the curve
probas = bst.predict(dtest)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(target_test, probas)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(target_test, probas)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])    
plt.figure()
plt.plot(fpr[1], tpr[1], label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show() 

#printing the accuracy, precision and recall scores for the model
print('Accuracy:{0:.2f}'.format(accuracy_score(target_test, y_test_preds)))
print('Precision:{0:.2f}'.format(precision_score(target_test, y_test_preds)))
print('Recall:{0:.2f}'.format(recall_score(target_test, y_test_preds)))
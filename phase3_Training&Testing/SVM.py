#SVM SVC classifier
from sklearn import preprocessing
from scipy import interp
import pylab as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    
##loading the datasets
target = (np.load('target_1980_2010.npy').astype(float))
data = (np.load('data_selected_1980_2010.npy').astype(float))
target = target[:, 1:]
# setting seed to reproduce results
seed =123
# Normalizing the predictors
N_data = preprocessing.normalize(data)
#splitting the train and test data & training the model for prediction
data_train, data_test, target_train, target_test = train_test_split(N_data, target, test_size=1803, random_state=seed)

# using SVM to train and predict the model
clf = svm.SVC(kernel='linear', C=26, class_weight={1: 14.4}, probability=True, random_state=seed)
clf.fit(data_train, target_train)
predictions = clf.predict(data_test)

#calcualte confusion matrix for the prediction
cnf_matrix = confusion_matrix(target_test, predictions)
plot_confusion_matrix(cnf_matrix, ['0', '1'])

print('Accuracy:{0:.2f}'.format(accuracy_score(target_test, predictions)))
print('Precision:{0:.2f}'.format(precision_score(target_test, predictions)))
print('Recall:{0:.2f}'.format(recall_score(target_test, predictions)))

probabilities = clf.predict_proba(data_test)[:,1]
# Compute ROC curve and area the curve
fpr, tpr, _ = roc_curve(target_test, probabilities)
roc_auc = auc(fpr, tpr)  
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for the test data')
plt.legend(loc="lower right")
plt.show() 

#10 fold cross validation and ROC curves
target = np.reshape(target,[11300,])
cv = StratifiedKFold(n_splits = 10, random_state=seed)

# generating roc curves for different folds
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(N_data, target):
    probas_ = clf.fit(N_data[train], target[train]).predict_proba(N_data[test])
    # Compute ROC curve and area the curve for different folds
    fpr, tpr, thresholds = roc_curve(target[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
    
pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
pl.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

pl.xlim([-0.05, 1.05])
pl.ylim([-0.05, 1.05])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('ROC curve for 10 fold cross validation')
pl.legend(loc="lower right")
pl.show()  
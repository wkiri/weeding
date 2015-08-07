#!/usr/bin/env python
# eval_classifier.py
# Read in the pre-processed Wesleyan data set (see process_data.py)
# and train and evaluate classifiers to predict weeding decisions.

import sys, os
import pickle  # to read in
import datetime
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn import tree
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from scipy.stats import chi2_contingency
from sklearn.feature_selection import chi2
from collections import Counter
from sklearn.externals.six import StringIO  
#import pydot

infile  = 'wesleyan.pkl'
resfile = 'results.pkl'
tau_values = np.linspace(0,1,11)

# weeding efficiency
def weeding_efficiency(cm):
    # e_w = a(a+b+c+d) / ((a+b)(a+c))
    # Note that this assumes a contingency matrix with 
    # Withdraw first and Keep second (it is weeding efficiency).
    # Since ours are alphabetical, we have to invert.
    # Also assumes predicted = rows and human = columns, 
    # which is also inverted.  Argh.
    # e_w = d(a+b+c+d) / ((b+d)(c+d))
    e_w = cm[1,1]*np.sum(cm)*1.0 / (cm[0,1]+cm[1,1]) / (cm[1,0]+cm[1,1])

    return e_w
    

def train_and_eval(clf, train, test, labels_train, labels_test):

    clf.fit(train, labels_train)

    # Predict on training data, then test data
    pred_tr = clf.predict(train)

    cm = confusion_matrix(labels_train, pred_tr)
    print cm
    print 'Training accuracy: %d / %d = %.2f%%' % (cm[0,0] + cm[1,1], 
                                                   np.sum(cm),
                                                   (cm[0,0] + cm[1,1])*100.0\
                                                       / np.sum(cm))
    # Report phi coefficient (also known as Matthew's correlation coeff)
    phi = matthews_corrcoef(labels_train, pred_tr)
    print 'Phi: %.2f' % phi,
    # Significance - use chi-2 with 1 dof: chi2 = N * phi^2
    print 'Sig (chi^2): %.2f' % (len(labels_train) * pow(phi,2)),
    print 'E_W = %.2f' % weeding_efficiency(cm)
    '''
    print 'Alternatively...'
    print chi2_contingency(cm)
    # need to convert these to ints to use chi2()
    pred_tr = np.reshape(pred_tr, (pred_tr.size,1))
    labels_train = np.reshape(labels_train, (labels_train.size,1))
    print chi2(pred_tr, labels_train)
    '''

    ######## Testing data ########
    pred_te = clf.predict(test)

    cm = confusion_matrix(labels_test, pred_te)
    print cm
    print 'Testing accuracy: %d / %d = %.2f%%' % (cm[0,0] + cm[1,1], 
                                                   np.sum(cm),
                                                   (cm[0,0] + cm[1,1])*100.0\
                                                       / np.sum(cm))
    # Report phi coefficient (also known as Matthew's correlation coeff)
    phi = matthews_corrcoef(labels_test, pred_te)
    print 'Phi: %.2f' % phi,
    # Significance - use chi-2 with 1 dof: chi2 = N * phi^2
    print 'Sig (chi^2): %.2f' % (len(labels_test) * pow(phi,2)),

    print 'Alternatively...'
    try:
        print chi2_contingency(cm)[1]
    except:
        print "couldn't compute chi2 sig."
    '''
    # need to convert these to ints to use chi2()
    pred_te = np.reshape(pred_te, (pred_te.size,1))
    labels_test = np.reshape(labels_test, (labels_test.size,1))
    try:
        print chi2(pred_te, labels_test)
    except:
        pass
    '''
    print 'E_W = %.2f' % weeding_efficiency(cm)

    # Sweep a confidence threshold
    try:
        conf_te = np.max(clf.predict_proba(test),axis=1)
    except:
        conf_te = -1
        return clf

    res        = np.zeros((len(tau_values),8))
    total_weed = len([l for l in labels_test if l == 'Withdrawn'])
    for i,tau in enumerate(tau_values):
        conf_pred = [(l,p) for (l,p,c) in zip(labels_test, pred_te, conf_te) \
                         if c >= tau]
        if len(conf_pred) == 0:
            res[i,:] = 0
            continue

        l, p = zip(*conf_pred)
        cm = confusion_matrix(l, p)
        res[i,0] = tau
        res[i,1] = (cm[0,0] + cm[1,1])*100.0 / np.sum(cm) # accuracy
        res[i,2] = cm[1,1]*100.0 / total_weed             # recall
        res[i,3] = cm[1,1]*100.0 / (cm[0,1] + cm[1,1])    # precision
        res[i,4] = weeding_efficiency(cm)  # efficiency
        res[i,5] = matthews_corrcoef(l, p) # phi
        try:
            res[i,6] = chi2_contingency(cm)[1]
        except:
            res[i,6] = np.nan
        res[i,7] = np.sum(cm) # number of items used

    print res

    return (clf, res)


def eval_baseline(labels, pred):

    cm = confusion_matrix(labels, pred)
    print cm

    print 'Accuracy: %d / %d = %.2f%%' % (cm[0,0] + cm[1,1], np.sum(cm),
                                          (cm[0,0] + cm[1,1]) * 100.0 / np.sum(cm))

    # Report phi coefficient (also known as Matthew's correlation coeff)
    phi = matthews_corrcoef(labels, pred)
    print 'Phi: %.2f' % phi,
    # Significance - use chi-2 with 1 dof: chi2 = N * phi^2
    print 'Sig (chi^2): %.2f' % (len(labels) * pow(phi,2))
    print 'Alternatively...'
    try:
        print chi2_contingency(cm)[1]
    except:
        print "couldn't compute chi2 sig."
    print 'Recall = %.2f'    % (cm[1,1]*100.0 / (cm[1,0] + cm[1,1]))
    print 'Precision = %.2f' % (cm[1,1]*100.0 / (cm[0,1] + cm[1,1]))
    print 'E_W = %.2f' % weeding_efficiency(cm)
    

# Read in pickled data file
with open(infile, 'r') as inf:
    (data, ids, labels) = pickle.load(inf)

print 'Read %d items with %d features.' % data.shape

# Ensure data is 2D
if len(data.shape) == 1:
    data = np.reshape(data, (data.size,1))

N,d = data.shape

# Split 50% train/test 
random.seed(0) # reproducibility
inds   = range(N)
random.shuffle(inds)
half_N = int(0.5*N)
train = inds[0:half_N]
test  = inds[half_N:-1]

# Shift/scale data to be 0-mean, 1-std dev (esp. for SVM)
scaler = StandardScaler().fit(data[train])
data = scaler.transform(data)

# Predict same value for all test items
for p in ['Withdrawn', 'Keep']:
    print 'Baseline (%s):' % p
    pred = np.array([p] * len(labels[test]))
    eval_baseline(labels[test], pred)
    print
print

result = {}

# 1. Linear SVM
# Use dual=False when n_samples > n_features.
# Use random_state to seed the random number generator (reproducible).
print 'SVM (even slower with probabilities!):'
#clf = LinearSVC(dual=False, random_state=0)
clf = SVC(kernel='linear', random_state=0, probability=True) # sloooow!
print 'All features:'
(clf, result['SVM']) = train_and_eval(clf, data[train], data[test], 
                                      labels[train], labels[test])
#print '\nAll features but votes:'
#clf = train_and_eval(clf, data[train,0:4], data[test,0:4], 
#                     labels[train], labels[test])
print

# 1a. RBF SVM
# Use random_state to seed the random number generator (reproducible).
print 'RBF SVM (slow!):'
clf = SVC(kernel='rbf', random_state=0, probability=True)
print 'All features:'
(clf, result['SVM RBF']) = train_and_eval(clf, data[train], data[test], 
                                          labels[train], labels[test])
#print '\nAll features but votes:'
#clf = train_and_eval(clf, data[train,0:4], data[test,0:4], 
#                     labels[train], labels[test])
print

# 2. K-nearest-neighbor 
#for n_neighbors in [3,5,7,9,11]:
for n_neighbors in [3]:
    print '%d-nearest neighbor:' % n_neighbors
    clf = neighbors.KNeighborsClassifier(n_neighbors)
    print 'All features:'
    (clf, result['%d-NN' % n_neighbors]) = \
        train_and_eval(clf, data[train], data[test], 
                       labels[train], labels[test])
    #print '\nAll features but votes:'
    #clf = train_and_eval(clf, data[train,0:4], data[test,0:4], 
    #                     labels[train], labels[test])
    print

# 3. Gaussian Naive Bayes
print 'Gaussian Naive Bayes:'
clf = GaussianNB()
print 'All features:'
(clf, result['NB']) = train_and_eval(clf, data[train], data[test], 
                                     labels[train], labels[test])
#print '\nAll features but votes:'
#clf = train_and_eval(clf, data[train,0:4], data[test,0:4], 
#                     labels[train], labels[test])
print

# 4. Decision tree
print 'Decision tree:'
clf = tree.DecisionTreeClassifier()
print 'All features:'
(clf, result['DT']) = train_and_eval(clf, data[train], data[test], 
                                     labels[train], labels[test])
#print '\nAll features but votes:'
#clf = train_and_eval(clf, data[train,0:4], data[test,0:4], 
#                     labels[train], labels[test])
#print '\nFaculty votes alone:'
#clf = train_and_eval(clf, data[:,4], labels)
#print '\nLibrarian votes alone:'
#clf = train_and_eval(clf, data[:,5], labels)
print

# 5. Random Forest
print 'Random Forest:'
clf = RandomForestClassifier()
print 'All features:'
(clf, result['RF']) = train_and_eval(clf, data[train], data[test], 
                                     labels[train], labels[test])
#print '\nAll features but votes:'
#clf = train_and_eval(clf, data[train,0:4], data[test,0:4], 
#                     labels[train], labels[test])
#print '\nFaculty votes alone:'
#clf = train_and_eval(clf, data[:,4], labels)
#print '\nLibrarian votes alone:'
#clf = train_and_eval(clf, data[:,5], labels)
print

'''
# 6. AdaBoost
print 'AdaBoost:'
clf = AdaBoostClassifier()
print 'All features:'
clf = train_and_eval(clf, data, labels)
print '\nAll features but votes:'
clf = train_and_eval(clf, data[:,0:4], labels)
'''

# Save to pickled file
with open(resfile, 'w') as outf:
    pickle.dump(result, outf)

# Don't do this - eats up CPU and RAM and disk!
#dot_data = StringIO() 
#tree.export_graphviz(clf, out_file=dot_data) 
#graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#graph.write_pdf("dtree-weed.pdf") 

#!/usr/bin/env python
# eval_classifier.py
# Read in the pre-processed Wesleyan data set (see process_data.py)
# and train and evaluate classifiers to predict weeding decisions.

import sys, os
import pickle  # to read in
import datetime
import numpy as np
import random
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn import tree
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import chi2
from collections import Counter
from operator import itemgetter
#from sklearn.externals.six import StringIO  
#import pydot

infile  = 'wesleyan.pkl'
resfile = 'results.pkl'
# Start at 0.5
#tau_values = np.linspace(0.5,1,101)
tau_values = [0.5]


# Compute Yule's Q coefficient and statistical significance
# at the 0.01 level.
# Warning! Invalid if any cell is zero; shouldn't be used if 
# any cell < 5.
def yule_q(cm):
    # If any cell has fewer than 5 entries, return Inf
    if len(cm) != 2 or np.any(cm < 5):
        return (np.inf, False)

    # yule_q = (ad-bc)/(ad+bc) = (OR-1)/(OR+1)
    # This is symmetric, so I'm not worried about whether
    # week or keep comes first
    yq = (cm[0,0]*cm[1,1] - cm[0,1]*cm[1,0]) / \
        np.float(cm[0,0]*cm[1,1] + cm[0,1]*cm[1,0])

    # Stat sig comes from chi^2
    chi2 = yq / (np.sqrt(0.25 * np.power((1.0-np.power(yq,2)),2) * \
                             (1.0/cm[0,0] + 1.0/cm[0,1] + 
                              1.0/cm[1,0] + 1.0/cm[1,1])))
    if chi2 > 2.33: # alpha = 0.01
        sig = True
    else:
        sig = False

    return (yq, sig)


# weeding efficiency = ratio of classifier precision to the original
# (label) precision
def weeding_efficiency(cm, orig_prec, one_pred):
    # Assuming rows = machine and cols = human:
    # classifier precision: d / (c+d)
    if len(cm) > 1:
        if (cm[1,0]+cm[1,1]) > 0:
            e_w = cm[1,1]*1.0/(cm[1,0]+cm[1,1]) /  orig_prec
        else: # all 'Keep'
            e_w = 0.0 / orig_prec
    else:
        # Only one cell in cm, meaning full agreement.
        # Precision depends on whether it's all 'Keep' or all 'Withdraw'.
        if one_pred == 'Withdrawn':
            e_w = 1.0 / orig_prec
        else: # all 'Keep'
            e_w = 0.0 / orig_prec

    return e_w
    

def train_and_eval(clf, train, test, labels_train, labels_test):

    clf.fit(train, labels_train)

    # Predict on training data, then test data
    pred_tr = clf.predict(train)

    # Confusion matrix rows = first arg, cols = second arg
    cm = confusion_matrix(pred_tr, labels_train)
    print cm
    print 'Training accuracy: %d / %d = %.2f%%' % (cm[0,0] + cm[1,1], 
                                                   np.sum(cm),
                                                   (cm[0,0] + cm[1,1])*100.0\
                                                       / np.sum(cm))
    # Report phi coefficient (also known as Matthew's correlation coeff)
    phi = matthews_corrcoef(labels_train, pred_tr)
    print 'Phi: %.2f' % phi,
    # Significance - use chi-2 with 1 dof: chi2 = N * phi^2
    print 'Sig (chi^2): %.2f' % (len(labels_train) * pow(phi,2))

    (yq, sig) = yule_q(cm)
    print "Yule's Q: %.2f" % yq

    total_weed = len([l for l in labels_train if l == 'Withdrawn'])

    print 'E_W = %.2f' % weeding_efficiency(cm,
                                            total_weed*1.0/len(labels_train),
                                            pred_tr[0])


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
    print Counter(pred_te)

    cm = confusion_matrix(pred_te, labels_test)
    print cm
    print 'Testing accuracy: %d / %d = %.2f%%' % (cm[0,0] + cm[1,1], 
                                                   np.sum(cm),
                                                   (cm[0,0] + cm[1,1])*100.0\
                                                       / np.sum(cm))
    # Report phi coefficient (also known as Matthew's correlation coeff)
    phi = matthews_corrcoef(labels_test, pred_te)
    print 'Phi: %.2f' % phi,
    # Significance - use chi-2 with 1 dof: chi2 = N * phi^2
    sig = len(labels_test) * pow(phi,2)
    print 'Sig (chi^2): %.2f' % sig,
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
    # Report Yule's Q for agreement
    (yq, sig_yq) = yule_q(cm)
    print "Yule's Q: %.2f" % yq

    print 'E_W = %.2f' % weeding_efficiency(cm,
                                            total_weed*1.0/len(labels_test),
                                            pred_te[0])

    # Sweep a confidence threshold
    try:
        # Get the max confidence classification
        conf_te = np.max(clf.predict_proba(test), axis=1)
        # Get the confidence in the "Withdrawn" class (class 1)
        #conf_te = clf.predict_proba(test)[:,1]
    except:
        conf_te = -1
        return clf

    res        = np.ones((len(tau_values),10))*np.nan
    total_weed = len([l for l in labels_test if l == 'Withdrawn'])
    for i,tau in enumerate(tau_values):
        res[i,0] = tau

        conf_pred = [(l,p) for (l,p,c) in zip(labels_test, pred_te, conf_te) \
                         if c >= tau]
        if len(conf_pred) == 0:
            continue

        l, p = zip(*conf_pred)
        # Pretend that these are all predicted to be "withdrawn"
        #p = ['Withdrawn'] * len(p)
        #print Counter(p)
        cm = confusion_matrix(p, l)
        #print tau, cm
        if tau == 1.0:
            print tau, cm
        if len(cm) > 1:
            res[i,1] = (cm[0,0] + cm[1,1])*100.0 / np.sum(cm) # accuracy
            res[i,2] = cm[1,1]*100.0 / total_weed             # recall
        else:
            # Only one cell in cm, meaning full agreement.
            res[i,1] = 1.0 # accuracy
            # Recall and precision depend on 
            # whether it's all 'Keep' or all 'Withdraw'.
            if p[0] == 'Withdrawn':
                res[i,2] = cm[0,0]*100.0 / total_weed # recall
                res[i,3] = 1.0
            else: # all 'Keep'
                res[i,2] = 0.0 # recall
                res[i,3] = 0.0 # precision 
        try:
            if cm[1,1] == 0: # no accurate withdraw predictions
                res[i,3] = 0.0 # precision
            else:
                res[i,3] = cm[1,1]*100.0 / (cm[1,0] + cm[1,1])    # precision
        except:
            pass
        res[i,4] = weeding_efficiency(cm, 
                                      total_weed*1.0/len(labels_test),
                                      p[0]) # efficiency
        

        res[i,5] = matthews_corrcoef(l, p) # phi
        try:
            #res[i,6] = 1 - stats.chi2.cdf(sum(sum(cm))*pow(res[i,5],2),1)
            res[i,6] = len(l) * pow(phi,2)
            print res[i,6]
            #res[i,6] = chi2_contingency(cm)[1]
        except:
            pass

        # Report Yule's Q for agreement
        (res[i,7], res[i,8]) = yule_q(cm)

        res[i,9] = np.sum(cm) # number of items used

    #print res

    return (clf, res)


def eval_baseline(labels, pred):

    cm = confusion_matrix(pred, labels)
    print cm

    acc = (cm[0,0] + cm[1,1]) * 100.0 / np.sum(cm)
    print 'Accuracy: %d / %d = %.2f%%' % (cm[0,0] + cm[1,1], np.sum(cm), acc)

    # Report phi coefficient (also known as Matthew's correlation coeff)
    phi = matthews_corrcoef(labels, pred)
    print 'Phi: %.2f' % phi,
    # Significance - use chi-2 with 1 dof: chi2 = N * phi^2
    sig = len(labels) * pow(phi,2)
    print 'Sig (chi^2): %.2f' % sig
    '''
    print 'Alternatively...'
    try:
        print chi2_contingency(cm)[1]
    except:
        print "couldn't compute chi2 sig."
    '''

    # Report Yule's Q for agreement
    (yq, sig_yq) = yule_q(cm)
    print "Yule's Q: %.2f" % yq

    recall = (cm[1,1]*100.0 / (cm[0,1] + cm[1,1]))
    if (cm[1,0] + cm[1,1]) > 0:
        precision = (cm[1,1]*100.0 / (cm[1,0] + cm[1,1]))
    else:
        precision = 0.0

    print 'Recall = %.2f'    % recall
    print 'Precision = %.2f' % precision

    total_weed = len([l for l in labels if l == 'Withdrawn'])
    eff = weeding_efficiency(cm,
                             total_weed*1.0/len(labels),
                             pred[0])
    print 'E_W = %.2f' % eff

    # Accuracy, recall, precision, efficiency, phi, sig, yule-q, sig, #items
    return (acc, recall, precision, eff, phi, sig, yq, sig_yq, np.sum(cm))


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
print 'data mean:'
print scaler.mean_
print 'data std:'
print scaler.std_
sys.exit(0)
data = scaler.transform(data)

# Load previously saved results
if os.path.exists(resfile):
    result = pickle.load(open(resfile, 'r'))
else:
    result = {}

# Predict same value for all test items
for p in ['Withdrawn', 'Keep']:
    print 'Baseline (%s):' % p
    pred = np.array([p] * len(labels[test]))
    # Rename this baseline 
    if p == 'Withdrawn':
        p = 'Withdraw'
    result[p] = eval_baseline(labels[test], pred)
    print
print

# Save results to pickled file
with open(resfile, 'w') as outf:
    pickle.dump(result, outf)

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

# 1. Linear SVM #################################################
print 'Linear SVM (even slower with probabilities!):'
'''
param_dist = {'C': np.logspace(-10,1,12)}
# Note: can give class weights with class_weight={'Keep':1,'Withdrawn':2}
clf = SVC(kernel='linear', random_state=0)
grid_search = GridSearchCV(clf, param_grid=param_dist, n_jobs=-1)
start = time()
grid_search.fit(data[train], labels[train])
print("GridSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), len(param_dist['C'])))
report(grid_search.grid_scores_)
clf    = grid_search.best_estimator_
best_C = clf.C

# Use dual=False when n_samples > n_features.
# Use random_state to seed the random number generator (reproducible).
#clf = LinearSVC(dual=False, random_state=0)
clf = SVC(kernel='linear', C=best_C, random_state=0, probability=True) # sloooow!
print 'All features:'
'''

clf = pickle.load(open('models/SVM.pkl'))
(clf, result['SVM']) = train_and_eval(clf, data[train], data[test], 
                                      labels[train], labels[test])
# Save out the trained classifier
#pickle.dump(clf, open('models/SVM.pkl', 'w'))
print

# Save results to pickled file
with open(resfile, 'w') as outf:
    pickle.dump(result, outf)

# 1a. RBF SVM ###############################################
# Use random_state to seed the random number generator (reproducible).
print 'RBF SVM (slow!):'
'''
# Param search for RBF SVM takes way too long (data set is too large?)
# especially for large C
# so just use the best_C from above
# and try three gammas
#param_dist = {'C': np.logspace(-3, 10, 14),
param_dist = {'C': np.logspace(-10, 1, 11),
              'gamma': np.logspace(-2, 3, 6)}
# Note: can give class weights with class_weight={'Keep':1,'Withdrawn':2}
clf = SVC(kernel='rbf', random_state=0)
# Does 3-fold CV
#n_iter_search = 50
#my_search = RandomizedSearchCV(clf, param_distributions=param_dist, 
#                                   n_iter=n_iter_search)
my_search = GridSearchCV(clf, param_grid=param_dist, n_jobs=-1)
start = time()
my_search.fit(data[train], labels[train])
n_iter_search = len(param_dist['gamma']) * len(param_dist['C'])
print("SearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(my_search.grid_scores_)
clf    = my_search.best_estimator_
best_C = clf.C
best_gamma = clf.gamma

clf = SVC(kernel='rbf', C=best_C, gamma=best_gamma,
          random_state=0, probability=True)
print 'All features:'
'''
clf = pickle.load(open('models/SVM-RBF.pkl'))
(clf, result['SVM RBF']) = train_and_eval(clf, data[train], data[test], 
                                          labels[train], labels[test])
# Save out the trained classifier
#pickle.dump(clf, open('models/SVM-RBF.pkl', 'w'))
print

# Save results to pickled file
with open(resfile, 'w') as outf:
    pickle.dump(result, outf)


# 2. K-nearest-neighbor ###########################################
'''

# Select best k
n_iter_search = 100
param_dist = {'n_neighbors': stats.randint(1,200)}

clf = neighbors.KNeighborsClassifier()
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
start = time()
random_search.fit(data[train], labels[train])
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)
clf    = random_search.best_estimator_

print 'All features:'
'''
clf = pickle.load(open('models/NN.pkl'))
best_k = clf.n_neighbors
print '%d-nearest neighbor:' % best_k

(clf, result['%d-NN' % best_k]) = \
    train_and_eval(clf, data[train], data[test], 
                   labels[train], labels[test])
# Save out the trained classifier
#pickle.dump(clf, open('models/NN.pkl', 'w'))
print

# Save results to pickled file
with open(resfile, 'w') as outf:
    pickle.dump(result, outf)

# 3. Gaussian Naive Bayes #########################################
print 'Gaussian Naive Bayes:'
'''
clf = GaussianNB()
'''
print 'All features:'
clf = pickle.load(open('models/NB.pkl'))
(clf, result['NB']) = train_and_eval(clf, data[train], data[test], 
                                     labels[train], labels[test])
# Save out the trained classifier
#pickle.dump(clf, open('models/NB.pkl', 'w'))
print

# Save results to pickled file
with open(resfile, 'w') as outf:
    pickle.dump(result, outf)

# 4. Decision tree ################################################
print 'Decision tree:'
'''
n_iter_search = 100
param_dist = {'max_depth': [3,5,None],
              'max_features': stats.randint(1,8),
              'criterion': ['gini', 'entropy']}

clf = tree.DecisionTreeClassifier()
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
start = time()
random_search.fit(data[train], labels[train])
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)
clf    = random_search.best_estimator_

print 'All features:'
'''
clf = pickle.load(open('models/DT.pkl'))
(clf, result['DT']) = train_and_eval(clf, data[train], data[test], 
                                     labels[train], labels[test])
# Save out the trained classifier
#pickle.dump(clf, open('models/DT.pkl','w'))

# Visualize the trained decision tree as PDF 
dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data, 
                     feature_names=['age','n_checkouts',
                                    'uslibs','peerlibs',
                                    'hathicopy','hathipub',
                                    'facultykeep','librariankeep']) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("dtree-weed.pdf") 
print

# Save results to pickled file
with open(resfile, 'w') as outf:
    pickle.dump(result, outf)


# 5. Random Forest #############################################
print 'Random Forest:'
'''
n_iter_search = 50
param_dist = {'max_depth': [3,5,None],
              'max_features': stats.randint(1,8),
              'criterion': ['gini', 'entropy'],
              'n_estimators': [10,20,50,100,500]}
clf = RandomForestClassifier()
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
start = time()
random_search.fit(data[train], labels[train])
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)
clf    = random_search.best_estimator_
'''
clf = pickle.load(open('models/RF.pkl'))
(clf, result['RF']) = train_and_eval(clf, data[train], data[test], 
                                     labels[train], labels[test])
# Save out the trained classifier
#pickle.dump(clf, open('models/RF.pkl', 'w'))

# Save results to pickled file
with open(resfile, 'w') as outf:
    pickle.dump(result, outf)


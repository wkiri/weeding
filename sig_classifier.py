#!/usr/bin/env python
# eval_classifier.py
# Read in saved evaluation results and report statistical significance.

import sys, os
import pickle  # to read in
import numpy as np
import math
import random
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import chi2
from collections import Counter
from operator import itemgetter

resfile = 'results-overall.pkl'

# Load previously saved results
if os.path.exists(resfile):
    result = pickle.load(open(resfile, 'r'))
else:
    print 'No %s found.  Run eval_classifier.py.' % resfile

# 1. Individual significance

print 'Are these values significant?'
# This formula is from Pearson's chi-squared test on wikipedia
# chi-2: N (p_correct ((n_correct/N - p_correct)/p_correct)^2 +
#           p_incorrect ((n_incorrect/N - p_incorrect)/p_incorrect)^2)

# p-value table:
# https://en.wikipedia.org/wiki/Chi-squared_distribution#Table_of_.CF.872_value_vs_p-value
# "The p-value is the probability of observing a test statistic at least as 
# extreme in a chi-squared distribution." ( smaller = more significant )
# Here, assume 1 dof.

def chi2(N, p1, obs1):
    outcome_1 = p1 * math.pow((obs1 - p1)/p1, 2)
    outcome_2 = (1-p1) * math.pow(((1-obs1) - (1-p1))/(1-p1), 2)
    return N * (outcome_1 + outcome_2)

'''
for c in result.keys():
    if c in ['Keep','Withdraw']:
        acc  = result[c][0]
        rec  = result[c][1]
        prec = result[c][2]
        N    = result[c][8]
    else:
        acc  = result[c][0,1]
        rec  = result[c][0,2]
        prec = result[c][0,3]
        N    = result[c][0,9]

    obs_correct   = acc/100.0
    obs_recall    = rec/100.0
    obs_precision = prec/100.0

    # Accuracy: Assume default p_correct = 0.5 (flip a coin)
    p_correct = 0.5
    # chi2 tests if observed accuracy is significantly DIFFERENT
    # from random (not significantly BETTER)
    chi2_acc = chi2(N, p_correct, obs_correct)
    print 'For %s (N=%d), %.2f acc -> %f chi2' % (c, int(N),
                                                  acc, chi2_acc),
    # Wikipedia says cutoffs are 
    # 3.84 (p=0.05), 6.64 (p=0.01), 10.83 (p=0.001)
    if chi2_acc > 10.83:
        print '***' # sig at p<0.001
    else:
        print

    # Recall: Assume default E[recall] = 0.5
    exp_recall = 0.5
    chi2_recall = chi2(N, exp_recall, obs_recall)
    print 'For %s (N=%d), %.2f recall -> %f chi2' % (c, int(N),
                                                     rec, chi2_recall),
    # Wikipedia says cutoffs are 
    # 3.84 (p=0.05), 6.64 (p=0.01), 10.83 (p=0.001)
    if chi2_recall > 10.83:
        print '***' # sig at p<0.001
    else:
        print

    # Accuracy: Assume default E[precision] = 0.4
    exp_precision  = 0.4
    chi2_precision = chi2(N, exp_precision, obs_precision)
    print 'For %s (N=%d), %.2f precision -> %f chi2' % (c, int(N),
                                                        prec, chi2_precision),
    # Wikipedia says cutoffs are 
    # 3.84 (p=0.05), 6.64 (p=0.01), 10.83 (p=0.001)
    if chi2_precision > 10.83:
        print '***' # sig at p<0.001
    else:
        print
'''

'''
    # Significantly better: compare the difference between acc and random
    # assuming a normal distribution around random's performance
    # http://www.bmj.com/about-bmj/resources-readers/publications/statistics-square-one/6-differences-between-percentages-and
    n1 = obs_correct * N
    n2 = p_correct * N # expected by chance
    z = (n1 - n2) / math.sqrt(n1 + n2)
    print '%.2f z-score.' % z
'''

# 2. Pairwise difference significance

# Std error given the mean value for two populations of sizes N1 and N2
# See http://www.bmj.com/about-bmj/resources-readers/publications/statistics-square-one/6-differences-between-percentages-and
# Assumes values are in range 0 to 100 (not 0 to 1)
def std_err(mean_val, N1, N2):
    se = math.sqrt((mean_val*(100-mean_val))/N1 + 
                   (mean_val*(100-mean_val))/N2)
    return se

def print_sig(mean_val, diff, N1, N2, c1, c2):
    se = std_err(mean_val, N1, N2)
    z  = diff/se
    # z >= 1.96:  p = 0.05
    # z >= 2.576: p = 0.01
    if abs(z) > 2.576:
        print '**',
    elif abs(z) > 1.96:
        print '*',
        
    print 'For %s, %s: diff = %.2f, SE = %.2f, z = %.2f' % \
        (c1, c2, diff, se, z)


print
print 'Are these differences significant?'

vals = ['Accuracy', 'Recall', 'Precision']
for i, v in enumerate(vals):
    print '%s ------------' % v
    for c1 in result.keys():
        if c1 in ['Keep','Withdraw']:
            c1_val = result[c1][i]
            c1_N   = result[c1][8]
        else:
            c1_val = result[c1][0,i+1]
            c1_N   = result[c1][0,9]

            for c2 in result.keys():
                if c1 == c2: 
                    continue
                if c2 in ['Keep','Withdraw']:
                    c2_val = result[c2][i]
                    c2_N   = result[c2][8]
                else:
                    c2_val = result[c2][0,i+1]
                    c2_N   = result[c2][0,9]
                
                mean_val = (c1_val + c2_val)/2.0

                # From BMJ link, sig test for diff in two proportions
                # (this assumes Gaussian distribution of the *difference*
                # for null hypothesis of equal means... how many std devs
                # you are apart says how unlikely they are equal means).
                print_sig(mean_val, c1_val - c2_val, c1_N, c2_N, c1, c2)

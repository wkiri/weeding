#!/usr/bin/env python
# plot_results.py

import sys, os
import pickle  # to read in
import pylab
import numpy as np

resfile = 'results.pkl'

m = ['x','o','*','+','.']

# Read in pickled results file
with open(resfile, 'r') as inf:
    result = pickle.load(inf)
print 'Plotting results for',
print result.keys()

# Replace 0-precision with NaN
for c in result.keys():
    if c in ['Keep','Withdraw']:
        continue
    result[c][:,3] = np.where(result[c][:,3] == 0,
                              np.nan,
                              result[c][:,3])

# Replace 0-phi with NaN
for c in result.keys():
    if c in ['Keep','Withdraw']:
        continue
    result[c][:,5] = np.where(result[c][:,5] == 0,
                              np.nan,
                              result[c][:,5])

# Could use this to box or limit the sig results
print 'Phi Significance'
for c in result.keys():
    if c in ['Keep','Withdraw']:
        continue
    print c, result[c][:,6]

# Plot results as a fn of tau
ylims = {'Accuracy': [35,100],
         'Recall':   [0,100],
         'Precision':[0,100],
         'Efficiency':[0,3],
         'Phi':      [0,1],
         'Yule-Q':   [0.8,1]}
ls = {'Keep': 'dashdot',
      'Withdraw': 'dashed'}
for (ind, value) in enumerate(['Accuracy',
                               'Recall',
                               'Precision',
                               'Efficiency',
                               'Phi',
                               'Phi-Sig',
                               'Yule-Q',
                               'Yule-Q-Sig']):
    pylab.clf()
    i = 0
    if '-Sig' in value:
        continue
    for c in result.keys():
        if c in ['Keep','Withdraw']:
            continue
        # For Phi and Yule's Q, only plot significant ones
        if value == 'Phi' or value == 'Yule-Q':
            use = np.where(result[c][:,ind+2])
            pylab.plot(result[c][use,0][0], result[c][use,ind+1][0], 
                       label=c, marker=m[i%len(m)])
        else:
            pylab.plot(result[c][:,0], result[c][:,ind+1], 
                       label=c, marker=m[i%len(m)])
        
        i += 1
    if value != 'Phi' and value != 'Yule-Q':
        # Add baselines.  Use tau from NB.
        for b in ['Keep','Withdraw']:
            pylab.plot([result['NB'][0,0],
                        result['NB'][-1,0]],
                       [result[b][ind],
                        result[b][ind]],
                       label=b, linestyle=ls[b])

    pylab.legend(loc=4, prop={'size':11}) # fontsize=10)
    pylab.xlabel('Tau (confidence threshold)')
    pylab.ylabel(value)
    pylab.ylim(ylims[value])
    pylab.savefig('fig/tau-%s.pdf' % value)

# Plot ROC
pylab.clf()
i = 0
for c in result.keys():
    if c in ['Keep','Withdraw']:
        continue
    pylab.plot(result[c][:,2], result[c][:,3], label=c, marker=m[i%len(m)])
    i += 1

for b in ['Keep','Withdraw']:
    pylab.plot(result[b][1], result[b][2], label=b, marker=m[i%len(m)],
               markersize=10)
    i += 1

pylab.legend(loc=4, prop={'size':11}) #, fontsize=10)
pylab.xlabel('Recall')
pylab.ylabel('Precision')
pylab.savefig('fig/roc.pdf')

for b in ['Keep','Withdraw']:
    print b, result[b]

# Output latex table for tau=0.5 performance
print
print '\\begin{table}'
print "\caption{Test performance for automated methods of predicting weeding decisions. Yule's Q is a statistical measure of agreement.  Stat sig is True if Yule's Q is statistically significant for p=0.01.}"
print '\label{tab:res}'
print '\\begin{center}'
print '\\begin{tabular}{|l|c|cc|cc|} \hline'
print "Method & Accuracy & Recall & Precision & $\phi$ & Yule's Q \\\\ \hline"
for b in ['Keep','Withdraw']:
    print "Baseline (all ``%s'') & %.2f & %.2f & %.2f & N/A & N/A \\\\" % \
        (b, result[b][0], result[b][1], result[b][2])
print '\hline'
for c in result.keys():
    if c in ['Keep','Withdraw']:
        continue
    # Critical value for chi-2 for 1 dof at p=0.01 is 6.6349
    print '%10s & %.2f & %.2f & %.2f & %.2f%s & %.2f%s \\\\' % \
        (c, result[c][0,1], result[c][0,2], result[c][0,3],
         result[c][0,5], '*' if (result[c][0,6] > 6.6349) else '',
         result[c][0,7], '*' if (result[c][0,8] == 1.0) else '')
print '\hline'
print '\end{tabular}'
print '\end{center}'
print '\end{table}'

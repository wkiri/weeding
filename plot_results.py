#!/usr/bin/env python
# plot_results.py

import sys, os
import pickle  # to read in
import pylab
import numpy as np

resfile = 'results.pkl'

m = ['x','o','.','+']

# Read in pickled results file
with open(resfile, 'r') as inf:
    result = pickle.load(inf)

# Replace 0-precision with NaN
for c in result.keys():
    result[c][:,3] = np.where(result[c][:,3] == 0,
                              np.nan,
                              result[c][:,3])

# Plot baselines

# Plot results as a fn of tau
for (ind, value) in enumerate(['Accuracy',
                               'Recall',
                               'Precision',
                               'Efficiency',
                               'Phi',
                               'Significance']):
    pylab.clf()
    for (i,c) in enumerate(result.keys()):
        pylab.plot(result[c][:,0], result[c][:,ind+1], 
                   label=c, marker=m[i%len(m)])
    pylab.legend(loc=2) #, fontsize=10)
    pylab.xlabel('Tau (confidence threshold)')
    pylab.ylabel(value)
    pylab.savefig('fig/tau-%s.png' % value)

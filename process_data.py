#!/usr/bin/env python
# Read in the Wesleyan data set and pre-process it to a useful format.
# Save the result in a pickled file.

import sys, os
import csv     # to read in
import pickle  # to write out
import datetime
import numpy as np
import pylab
import seaborn as sns # pretty plots
from sklearn import preprocessing
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency

infile   = 'wesleyan-nl.csv'
outfile  = 'wesleyan.pkl'
isbnfile = 'ISBNs.csv'

months = ['Jan', 'Feb', 'Mar', 'Apr',
          'May', 'Jun', 'Jul', 'Aug',
          'Sep', 'Oct', 'Nov', 'Dec']

# Reference date for determining time since last checkout.
refdate = datetime.date(2014, 1, 1)
# Default shelftime assigned if # checkouts > 0 and last circ date is missing
default_missing_shelftime = (refdate - datetime.date(2003, 1, 1)).days
print 'Default shelftime for checked-out items without a checkout date:', 
print default_missing_shelftime


def quality_report(vals, name):
    print '%s:' % name

    # Strings: Generate a histogram of values
    if type(vals[0]) == str:
        # Make a dictionary to get counts for each value
        valdict = {}
        for v in vals:
            if v in valdict.keys():
                valdict[v] += 1
            else:
                valdict[v] = 1

        for v in valdict.keys():
            print ' ' + str(v) + ':\t%d (%.2f%%)' % \
                (valdict[v], valdict[v] * 100.0 / sum(valdict.values()))

    # Ints/floats: report min/max/mean
    elif (type(vals[0]) == int or 
          type(vals[0]) == float or 
          type(vals[0]) == np.int64):
        vals_use = [v for v in vals if v > -1]
        if len(vals_use) > 0:
            print ' Min: '       + str(min(vals_use)),
            print '\tMax: '      + str(max(vals_use)),
            print '\tMean: %2.f' % (sum(vals_use)*1.0/len(vals_use)),
        print '\tMissing: ' + str(sum([1 for v in vals if v == -1]))

    else:
        print ' Unhandled type %s' % type(vals[0])

    print


# Analyze the correlation of variable x to target y
# y is assumed to be of type String (nominal)
def correlation(x, y, varname):
    print varname
    # Two distinct values?  Do a chi-square correlation
    uniq = set(x)
    if len(uniq) == 2:
        yvals = set(y)
        print '\t\t\t\t' + '\t'.join(yvals)
        for u_x in uniq:
            print u_x, len(x[x == u_x]), \
                len(x[x == u_x])*100.0/len(x),'\t',
            for u_y in yvals:
                print '\t%d' % len(x[x[y == u_y]==u_x]),
            print '\t%.2f' % (len(x[x[y == 'Withdrawn']==u_x])*1.0/\
                                  len(x[x==u_x]))
            print
        x = np.reshape(x, (x.size,1))
        print 'Chi^2:',
        print chi2(x, y)
        #print 'Chi^2 from contingency matrix:',
        # This is slow 
        #cm = confusion_matrix([str(xval) for xval in x], y)
        # This raises an error
        #print  chi2_contingency(cm)

    else:
        '''
        # Ints/floats: create a box plot and report means
        pylab.clf()
        use = x > -1
        if len(use[0]) == 0:
            return
        #sns.violinplot(x, y, scale="width")
        #sns.violinplot(x=y[use], y=x[use])
        sns.boxplot(x=y[use], y=x[use])
        pylab.savefig('fig/boxplot-%s.png' % varname)
        '''

        # Ints/floats: create a histogram
        pylab.clf()
        #sns.distplot(x, rug=True)
        sns.distplot(x[x != -1])
        pylab.xlabel(varname)
        pylab.savefig('fig/hist-%s.png' % varname)

        # Ints/floats: split by outcome
        pylab.clf()
        keeps      = (y == 'Keep')      & (x != -1)
        withdraws  = (y == 'Withdrawn') & (x != -1)
        #sns.distplot(x[keeps], rug=True, color='g')
        #sns.distplot(x[withdraws], rug=True, color='r')
        sns.distplot(x[keeps], color='g', label='Keep')
        sns.distplot(x[withdraws], color='r', label='Withdraw')
        pylab.xlabel(varname)
        pylab.legend()
        pylab.savefig('fig/hist-%s-split.png' % varname)


'''
# Read in the ISBN info
with open(isbnfile, 'r') as csvfile:
    rd = csv.reader(csvfile)

    # Parse the header
    try:
        line = rd.next()
        ind_id   = line.index('ITEM_ID')
        ind_isbn = line.index('ISBN')
    except ValueError:
        print 'Could not find one or more columns in ISBN file; check parsing.'
        sys.exit(1)

    # Items can have multiple ISBNs, one for each format,
    # so store a list: [isbn1, isbn2...]
    isbns = {}
    for line in rd:
        id   = int(line[ind_id])
        # Just take the ISBN, ignore comments after it
        isbn = line[ind_isbn].split()[0]
        if id in isbns.keys():
            isbns[id].append(isbn)
        else:
            isbns[id] = [isbn]
'''

# Read in original CSV file
with open(infile, 'r') as csvfile:
    rd = csv.reader(csvfile)

    # Parse the header
    try:
        line = rd.next()
        ind_id  = line.index('Local ITEM_ID')
        ind_en  = line.index('enumeration')
        ind_yr  = line.index('Publication Year')
        ind_dt  = line.index('Date of last checkout')
        ind_ch  = line.index('Checkouts since 1996')
        ind_us  = line.index('OCLC Holdings USA')
        ind_pr  = line.index('OCLC Holdings Peer')
        ind_hc  = line.index('Hathi In Copyright')
        ind_hp  = line.index('Hathi Public Domain')
        ind_fk  = line.index('facultykeep')
        ind_lk  = line.index('librariankeep')
        ind_dec = line.index('Wtihdrawn?')  # sic
    except ValueError:
        print 'Could not find one or more columns; check parsing.'
        sys.exit(1)

    data = []

    nlines = 0
    n_enums    = 0
    n_booksout = 0
    n_missdate = 0
    for line in rd:
        nlines += 1
        # Construct the feature vector
        # 0. Final decision (keep or withdraw/weed)
        dec = line[ind_dec]  # 'Withdrawn' or '' (Keep)
        if dec == '':
            dec = 'Keep'
        if dec not in ['Withdrawn', 'Keep']:
            print 'Unknown weeding decision: <%s>' % dec
            sys.exit(1)

        # If it's an enumeration, skip it
        en = line[ind_en]
        if en != '':
            n_enums += 1
            continue

        # 1. Item ID
        id = int(line[ind_id])

        # 2. Age (years since publication): integer >= 0
        age = refdate.year - int(line[ind_yr])
        if age < 0:
            print 'Error: age < 0 for pub date %s; setting to -1.' % line[ind_yr]
            age = -1

        # 3. Number of checkouts since 1996: integer >= 0
        n_checkout = int(line[ind_ch])
        #n_checkout = line[ind_ch]

        # 4. Days since last checkout: integer >= 0
        if line[ind_dt] == '': # date unspecified
            shelftime = -1     # or set to some max value? earlier than 1996.
        else:
            # Some are month/day/year.  
            # Some are day-month-year.
            try:
                # Try month/day/year first
                (m, d, y) = map(int, line[ind_dt].split('/'))
            except:
                try:
                    # Try day-month-year.
                    (dstr, mstr, ystr) = line[ind_dt].split('-')
                    m = months.index(mstr) + 1
                    d = int(dstr)
                    y = int(ystr)
                except:
                    print 'Error parsing m/d/y or d-m-y date from %s.' % \
                        line[ind_dt]
                    shelftime = -1
            # Years are specified with two digits and earliest should be 2003.
            # Sanity check:
            if (m < 1 or m > 12 or
                d < 1 or d > 31):
                print 'Error parsing m/d/y date from %s.' % line[ind_dt]
                continue
            if y > 14:       # I found at least one 1996
                y = 1900 + y
            elif y >= 0:
                y = 2000 + y
            else:
                print 'Error parsing m/d/y date from %s.' % line[ind_dt]
            shelftime = (refdate - datetime.date(y, m, d)).days

        # 5. Number of U.S. libraries holding a copy: integer >= 0
        n_uslib = int(line[ind_us])

        # 6. Number of peer libraries holding a copy: integer >= 0
        n_peerlib = int(line[ind_pr])

        # 7. Hathi in Copyright: boolean -> int
        hc = line[ind_hc]
        if hc == 'T' or hc == 'F':
            hathi_copy = int(bool(hc=='T'))
        else:
            hathi_copy = -1

        # 8. Hathi in public domain: boolean -> int
        hp = line[ind_hp]
        if hp == 'T' or hp == 'F':
            hathi_pub = int(bool(hp=='T'))
        else:
            hathi_pub = -1

        # 9. How many faculty members voted to keep it: integer >= 0
        n_fk = int(line[ind_fk])

        # 10. How many librarians voted to keep it: integer >= 0
        n_lk = int(line[ind_lk])
        #n_lk = line[ind_lk]

        # 11. Item appears in Resources for College Libraries (RCL): binary (0 = false, 1 = true)
        # Need to do a lookup.  Not sure how.  Get ISBN?
        # http://rclweb.net.libaccess.sjlibrary.org/
        # http://rclweb.net.libaccess.sjlibrary.org/AdvancedSearch/Index
        # http://rclweb.net.libaccess.sjlibrary.org/Search/Results?q=isbn%3d%5b0905169166%5d+&ast=pr


        # QC: if # checkouts == 0 but last checkout date is known,
        # update # checkouts to 1.  (This means that the book was
        # still checked out at the time the data was collected.)
        if n_checkout == 0 and shelftime != -1:
            n_checkout = 1
            n_booksout += 1

        # QC: if # checkouts > 0 but last checkout date is missing,
        # set last checkout date to 1/1/2003.
        if n_checkout > 0 and shelftime == -1:
            shelftime   = default_missing_shelftime
            n_missdate += 1
            continue  # Don't include these

        fv = [dec, id, age, n_checkout, shelftime, 
              n_uslib, n_peerlib, 
              hathi_copy, hathi_pub,
              n_fk, n_lk] 
        #print fv
        #raw_input()
        data += [fv]

    #--------- Data quality report ------------#
    print
    print 'Successfully parsed %d of %d items.' % (len(data), nlines)
    print ' %d were part of an enumeration (skipped).' % n_enums
    print ' %d were circulating (set # checkouts to 1).' % n_booksout
    print ' %d were checked out between 1996 and 2003 (set last circ date to 1/1/2003)' % n_missdate
    print

    (decs, ids, ages, n_checkouts, shelftime, 
     n_uslibs, n_peerlibs, hathi_copys, hathi_pubs, 
     n_fks, n_lks) = \
        zip(*data)
    quality_report(decs,        'Weeding decisions')
    quality_report(ages,        'Years since publication')
    quality_report(n_checkouts, 'Number of checkouts')
    quality_report(shelftime,   'Days since last checkout')
    quality_report(n_uslibs,    'Number of U.S. libraries with copy')
    quality_report(n_peerlibs,  'Number of peer libraries with copy')
    quality_report(hathi_copys, 'In Hathi w/copyright')
    quality_report(hathi_pubs,  'In Hathi public domain')
    quality_report(n_fks,       'Number of faculty votes')
    quality_report(n_lks,       'Number of librarian votes')

    # Transform data and labels into numpy array
    data   = np.array([ages, n_checkouts, shelftime, 
                       n_uslibs, n_peerlibs,
                       hathi_copys, hathi_pubs,
                       n_fks, n_lks]).T
    ids    = np.array(ids)
    labels = np.array(decs)

    #--------- Correlation analysis ------------#
    vals=range(len(labels))
    correlation(data[vals,0], labels[vals], 'ages')
    correlation(data[vals,1], labels[vals], 'n_checkouts')
    correlation(data[vals,2], labels[vals], 'shelftime')
    correlation(data[vals,3], labels[vals], 'n_uslibs')
    correlation(data[vals,4], labels[vals], 'n_peerlibs')
    correlation(data[vals,5], labels[vals], 'hathi_copys')
    correlation(data[vals,6], labels[vals], 'hathi_pubs')
    correlation(data[vals,7], labels[vals], 'n_fks')
    correlation(data[vals,8], labels[vals], 'n_lks')

    # Restrict data to features of interest
    #data = data[:,[0,2,7,8]]
    #data = np.reshape(data, (data.size,1))
    print data.shape

    # Save to pickled file
    with open(outfile, 'w') as outf:
        pickle.dump((data, ids, labels), outf)




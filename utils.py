from __future__ import division
import csv
from itertools import groupby

def begin_parse(input_file):
    '''String -> [String]. Take a filename for a tab-delimited
    file and get a list of the rows of the file.'''
    with open(input_file) as f:
        reader = csv.reader(f, delimiter = '\t')
        rows = [row for row in reader]
        return rows

def pluck(objects, attribute):
    '''Collection, String -> Collection. Like underscore.js's pluck.
    Gets the specified attribute from each item in the collection.'''
    return map(lambda obj: getattr(obj, attribute), objects)

def normalize_by_ur(candidates, sequence):
    '''[Candidate], [Number] -> [Number]. Each number in sequence belongs to one
    candidate, in order. Divides each number by the sum of all those numbers
    that are associated with the same UR. Useful for normalizing both empirical
    and predicted probabilities of candidates. A probability normalized by UR
    represents the probability that the given SR is the right output for its
    UR.'''
    labeled_items = zip(candidates, sequence)
    ur_sums = dict([(ur, sum([item for (cand, item) in group]))
          for (ur, group) # group is iterator over (candidate, item) pairs
          in groupby(labeled_items,
                     lambda (candidate, item): candidate.ur)])
    return [item / ur_sums[candidate.ur]
        for (candidate, item) in labeled_items]


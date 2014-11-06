'''OT-Py module for candidates.'''

from __future__ import division
import utils
from enum import Enum
import numpy as np
import collections

class Candidate:
    '''Object representing an input-output pair.'''
    def __init__(self, ur, sr, frequency, violations, hidden = None):
        '''ur: string, sr: string, frequency: float, violations: [int] -> None'''
        self.ur = ur # underlying form
        self.sr = sr # surface form
        self.full = hidden if hidden else sr # full structure if some of structure is hidden
        self.frequency = frequency
        self.violations = violations

    def __str__(self):
        '''None -> string. Candidates will be printed like "/ur/ ~ [sr]".'''
        return '/' + self.ur + '/ ~ [' + self.full + ']'

class TableauSet:
    '''Object representing a set of Candidates in one language.'''
    def __init__(self, tableau_file):
        '''tableau_file: string -> None. tableau_file is the name of a file in OT-Soft
        format.'''
        self.long_constraint_names = None
        self.short_constraint_names = None
        self.candidates = []
        self.violation_matrix = []
        self.frequency_array = []
        self.ur_bounds_full = []
        self.ur_bounds_sr = []
        self.sr_bounds = {}

        self.parse_tableau_file(tableau_file)
        self.empirical_distribution = self.get_empirical_probabilities()

    def parse_tableau_file(self, tableau_file, hidden = False):
        '''tableau_file: string, hidden: boolean -> ([string], [string], TableauSet).
        Parses tableau_file according to OT-Soft and OT-Help conventions. If
        hidden is True, the third column will be taken to contain hidden structures.

        Sets self.long_constraint_names, self.short_constraint_names,
        self.frequency_array, self.violation_matrix, self.ur_bounds_full,
        self.ur_bounds_sr'''
        self.hidden = hidden
        rows = utils.begin_parse(tableau_file)
        self.long_constraint_names = rows[0][3:]
        self.short_constraint_names = rows[1][3:]
        freq_index = 3 if hidden else 2
        header = 2

        # stateful: ur can be left out, in which case you use the most recent ur
        # if hidden is True, sr can also be left out in the same way
        ur = None
        sr = None
        frequency = None
        j = 0
        frequency_array = []
        violation_matrix = []
        ur_starts_full = []
        ur_starts_sr = []
        sr_starts = collections.defaultdict(list)
        for (i, row) in enumerate(rows[header:]):
            if row[0]: # new ur
                ur = row[0]
                ur_starts_full.append(i)
                ur_starts_sr.append(j)
            if row[1]: # new sr
                sr = row[1]
                frequency = row[freq_index]
                frequency_array.append(int(frequency))
                sr_starts[ur_starts_full[-1]].append(i)
                j += 1
            full = row[2] if hidden else sr
            violations = np.array([int(violation) if violation else 0
                for violation in row[freq_index + 1:]])
            # make violations negative
            violations = [-v if v >= 0 else v for v in violations]

            self.candidates.append(Candidate(ur, sr, frequency, violations))
            violation_matrix.append(violations)

        self.frequency_array = np.array(frequency_array)
        self.violation_matrix = np.array(violation_matrix)
        self.ur_bounds_full = self.close_bounds(ur_starts_full, len(rows) -
                header)
        self.ur_bounds_sr = self.close_bounds(ur_starts_sr, j)
        self.sr_bounds = dict([(key, self.close_bounds(values, len(rows) -
            header))
            for (key, values) in sr_starts.iteritems()])

    def close_bounds(self, starts, final_index):
        ends = starts + [final_index]
        return zip(starts, ends[1:])

    def normalize_by_ur(self, array, full = None):
        '''Works for 1D and 2D arrays where the first axis is candidates.'''
        full = self.hidden if full == None else full
        ur_bounds = self.ur_bounds_full if full else self.ur_bounds_sr
        chunks = []
        for (start, end) in ur_bounds:
            subarray = array[start:end]
            print 'subarray', subarray
            normalizer = np.sum(subarray, axis = 0)
            print 'normalizer', normalizer
            print 'division', subarray/normalizer
            chunks.append(subarray/normalizer)
        # reflatten
        return np.array([unit for chunk in chunks for unit in chunk])

    def get_empirical_probabilities(self):
        return self.normalize_by_ur(self.frequency_array, full = False)

    # def set_probabilities(self):
    #     frequencies = utils.pluck(self.candidates, 'frequency')
    #     probabilities = utils.normalize_by_ur(self.candidates, frequencies)
    #     for (candidate, probability) in zip(self.candidates, probabilities):
    #         candidate.probability = probability

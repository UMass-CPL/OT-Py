'''OT-Py module for candidates.'''

import utils
from enum import Enum
import numpy as np
import collections

class Candidate:
    '''Object representing an input-output pair.'''
    def __init__(self, ur, sr, frequency, violations):
        '''ur: string, sr: string, frequency: float, violations: [int] -> None'''
        self.ur = ur # underlying form
        self.sr = sr # surface form
        self.frequency = frequency
        self.violations = violations

    def __str__(self):
        '''None -> string. Candidates will be printed like "/ur/ ~ [sr]".'''
        return '/' + self.ur + '/ ~ [' + self.sr + ']'

class TableauSet:
    '''Object representing a set of Candidates in one language.'''
    def __init__(self, tableau_file):
        '''tableau_file: string -> None. tableau_file is the name of a file in OT-Soft
        format.'''
        (self.long_constraint_names,
        self.short_constraint_names,
        self.candidates) = self.parse_tableau_file(tableau_file)
        self.set_probabilities()

    def parse_tableau_file(self, tableau_file):
        '''tableau_file: string -> ([string], [string], TableauSet).
        Parses tableau_file according to OT-Soft and OT-Help conventions.'''
        rows = utils.begin_parse(tableau_file)
        long_constraint_names = rows[0][3:]
        short_constraint_names = rows[1][3:]
        candidates = []
        # # stateful: ur can be left out, in which case you use the most recent ur
        ur = None
        for row in rows[2:]:
            ur = row[0] if row[0] else ur
            sr = row[1]
            frequency = int(row[2]) if row[2] else 0
            violations = np.array([int(violation) if violation else 0
                for violation in row[3:]])
            violations = [-v if v >= 0 else v for v in violations]
            candidates.append(Candidate(ur, sr, frequency, violations))
        return long_constraint_names, short_constraint_names, candidates

    def set_probabilities(self):
        frequencies = utils.pluck(self.candidates, 'frequency')
        probabilities = utils.normalize_by_ur(self.candidates, frequencies)
        for (candidate, probability) in zip(self.candidates, probabilities):
            candidate.probability = probability

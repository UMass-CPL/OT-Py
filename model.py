'''OT-Py module for models of phonological grammar.'''

from sklearn.linear_model import LogisticRegression
from sklearn.learning_curve import learning_curve
import numpy as np
from scipy.stats import entropy as kldivergence # relative entropy is KL divergence
from scipy.optimize import minimize
import utils

    # needed methods:
    # def online_learn(probabilities):
    #     '''[Float] -> [(Int, [Float], [Float])]. Takes a list of probabilities
    #     and returns a list of records containing timestep, list of weights, and
    #     errors.'''

    # def get_typologies():
    #     '''None -> [[Candidate]]. Takes no arguments and returns a list of
    #     languages, that is, lists of winning candidates.'''


class MaximumEntropy():
    def __init__(self, tableau_set, constraints = None):
        self.tableau_set = tableau_set
        # these are constraint names - strings
        self.constraints = constraints if constraints else self.tableau_set.long_constraint_names
        self.model = LogisticRegression()
        self.violation_matrix = self.get_violation_matrix()

    def get_violation_matrix(self):
        '''None -> Numpy array of integers, with as many rows as candidates and as many columns as
        constraints.'''
        return np.array([candidate.violations
            for candidate in self.tableau_set.candidates])

    def batch_learn_categorical(self, categories):
        '''[Integer] -> [(String, Float)]. Takes a list of integers representing
        categories (probably only 0 and 1 for grammatical and ungrammatical).
        Returns a list a constraint names and their weights. Use 0 for
        grammatical in order to make weights behave as expected.'''
        self.model.fit(self.violation_matrix, categories)
        return zip(self.constraints, self.model.coef_[0])

    def batch_learn_variable(self, initial_weights = None):
        '''optional [Float] -> [(String, Float)]. Takes an optional list of probabilities,
        and returns a list of constraint names and their weights. Used for for
        finding constraint weights for data where candidates are associated with
        frequencies (a given UR maps to one SR so often, and another SR so
        often) rather than categories (this SR is grammatical and that one is
        ungrammatical).'''
        initial_weights = initial_weights if initial_weights else [0.0
                for c in range(len(self.constraints))]
        result = minimize(self.kl_objective, initial_weights)
        return zip(self.constraints, result.x)

    def kl_objective(self, weights):
        '''[Float] -> Float. Objective function for KL divergence minimization.
        Returns the KL divergence between the empirical probabilities of the
        candidates and the probabilities based on their calculated harmonies.'''
        empirical_distribution = [candidate.probability for candidate in
                self.tableau_set.candidates]
        # unnormalized predicted probability of every candidate
        scores = np.exp(np.dot(self.violation_matrix, weights))
        predicted_distribution = utils.normalize_by_ur(self.tableau_set.candidates, scores)
        return kldivergence(empirical_distribution, predicted_distribution)

    # def online_learn(self, probabilities = None):
    #     probs = probabilities if probabilities else [cand.probability for cand
    #             in self.tableau_set.candidates]
    #     (train_sizes_ab,
    #             train_scores,
    #             test_scores) = learning_curve(self.model, self.violation_matrix, probs) # TODO optional args
    #     return (train_sizes_ab, train_scores, test_scores)

    def get_distribution(self, weights):
        '''[Float] -> [(String, Float)]. Takes a list of weights, corresponding to
        constraints, and returns a list of pairs of candidate strings and their probabilities.'''
        self.model.coef_ = np.array([weights])
        self.model.intercept_ = 1
        cands = map(str, self.tableau_set.candidates)
        probs = self.model.predict_proba(self.violation_matrix)
        grammatical_probs, ungrammatical_probs = zip(*probs)
        return zip(cands, grammatical_probs)

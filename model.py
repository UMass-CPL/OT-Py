'''OT-Py module for models of phonological grammar.'''
from __future__ import division
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
        self.empirical_distribtion = self.tableau_set.get_empirical_probabilities()

    def get_candidate_tensor(self):
        return np.array([[[hidden for hidden in sr] for sr in ur] for ur in
            self.tableau_set.candidates])

    def batch_learn_categorical(self, categories):
        '''[Integer] -> [(String, Float)]. Takes a list of integers representing
        categories (probably only 0 and 1 for grammatical and ungrammatical).
        Returns a list a constraint names and their weights. Use 0 for
        grammatical in order to make weights behave as expected.'''
        self.model.fit(self.tableau_set.violation_matrix, categories)
        return zip(self.constraints, self.model.coef_[0])

    def batch_learn_variable(self, initial_weights = None):
        '''optional [Float] -> [(String, Float)]. Takes an optional list of probabilities,
        and returns a list of constraint names and their weights. Used for for
        finding constraint weights for data where candidates are associated with
        frequencies (a given UR maps to one SR so often, and another SR so
        often) rather than categories (this SR is grammatical and that one is
        ungrammatical).'''
        initial_weights = initial_weights if initial_weights else np.zeros(len(self.constraints))
        kl_grad = make_kl_gradient(self.tableau_set.empirical_distribution,
                                   self.tableau_set.violation_matrix,
                                   self.tableau_set.ur_bounds_full,
                                   self.tableau_set.sr_bounds)
        kl_obj = make_kl_objective(self.tableau_set.empirical_distribution,
                                   self.tableau_set.violation_matrix,
                                   self.tableau_set.normalize_by_ur,
                                   kl_grad)
        result = minimize(kl_obj, initial_weights, jac = True)
        return zip(self.constraints, result.x)

    # def online_learn(self, probabilities = None):
    #     probs = probabilities if probabilities else [cand.probability for cand
    #             in self.tableau_set.candidates]
    #     (train_sizes_ab,
    #             train_scores,
    #             test_scores) = learning_curve(self.model, self.violation_matrix, probs)
    #     return (train_sizes_ab, train_scores, test_scores)

    def get_distribution(self, weights):
        '''[Float] -> [(String, Float)]. Takes a list of weights, corresponding to
        constraints, and returns a list of pairs of candidate strings and their probabilities.'''
        self.model.coef_ = np.array([weights])
        self.model.intercept_ = 1
        cands = map(str, self.tableau_set.candidates)
        probs = self.model.predict_proba(self.tableau_set.violation_matrix)
        grammatical_probs, ungrammatical_probs = zip(*probs)
        return zip(cands, grammatical_probs)

# functions for optimization

def make_kl_objective(empirical_distribution, violations, normalizer, gradient_function):

    def kl_objective(weights):
        '''[Float] -> (Float, Function). Objective function for KL divergence minimization.
        Returns the KL divergence between the empirical probabilities of the
        candidates and the probabilities based on their calculated harmonies,
        and the function for finding the gradient of .'''
        scores = np.exp(np.dot(violations, weights))
        predicted_distribution = normalizer(scores)
        divergence = kldivergence(empirical_distribution, predicted_distribution)
        gradient = gradient_function(weights, predicted_distribution)
        return divergence, gradient

    return kl_objective

def make_kl_gradient(empirical_distribution, violations, ur_bounds, sr_bounds):

    def kl_gradient(weights, predicted_distribution):
        result = np.zeros(len(weights))
        for (start, end) in ur_bounds:
            ur_violations = violations[start:end]
            ur_probs = predicted_distribution[start:end]
            # should use ur_probs.T but numpy does it for you
            expected_violations_ur = np.dot(ur_probs, ur_violations)
            for (i, (sr_start, sr_end)) in enumerate(sr_bounds[start]):
                sr_violations = violations[sr_start:sr_end]
                sr_scores = predicted_distribution[sr_start:sr_end]
                # renormalize
                sr_probs = sr_scores / np.sum(sr_scores)
                expected_violations_sr = np.dot(sr_probs, sr_violations)
                expectation_difference = expected_violations_sr - expected_violations_ur
                # sum over all urs and srs
                result += empirical_distribution[i] * expectation_difference
        return -result

    return kl_gradient

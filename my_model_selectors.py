"""
TODO:
    * find BIC, DIC, and log-likelihood scoring methods in sklearn
"""

import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def optimal_num_states(self, optimizer=max):
        assert optimizer is max or optimizer is min

        scored_num_states = ((self.score(num_states), num_states)
                             for num_states
                             in range(self.min_n_components, self.max_n_components+1))
        valid_scored_num_states = [(s,n) for s,n in scores if s is not None]

        if not valid_scored_num_states:
            return None
        else:
            _, opt_num_states = optimizer(valid_scored_num_states)
            return opt_num_states

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        num_states = self.optimal_num_states(optimizer=min)
        return None if num_states is None else self.base_model(num_states)

    def score(self, num_states):
        try:
            model = self.base_model(num_states)
            logL, p, logN = model.score(self.X, self.lengths), num_states, math.log(len(self.X))
            bic_score = -2 * logL + p * logN
            return bic_score
        except ValueError:  # The hmmlearn library may not be able to train or score all models.
            return None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        num_states = self.optimal_num_states()
        return None if num_states is None else self.base_model(num_states)

    def score(self, num_states):
        try:
            model = self.base_model(num_states)
            M = len(self.words)
            log_prob_X = model.score(self.X, self.lengths)
            sum_log_prob_not_X = sum(model.score(*self.hwords[w]) for w in self.words if w != self.this_word)
            return log_prob_X - (1 / (M - 1)) * sum_log_prob_not_X
        except ValueError:  # The hmmlearn library may not be able to train or score all models.
            return None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        "() -> GaussianHMM"
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        num_states = self.optimal_num_states()
        return None if num_states is None else self.base_model(num_states)

    def score(self, num_states):
        "GaussianHMM -> float"
        try:
            model = self.base_model(num_states)
            log_likelihood_scores = []
            n_splits = min(3, len(self.sequences))
            split_method = KFold(n_splits=n_splits, random_state=self.random_state)

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                test_X, test_lengths =  combine_sequences(cv_test_idx, self.sequences)
                model.fit(train_X, train_lengths)
                split_score = model.score(test_X, test_lengths)
                log_likelihood_scores.append(split_score)

            average_log_likelihood = sum(log_likelihood_scores) / len(log_likelihood_scores)
            return average_log_likelihood
        except ValueError:  # The hmmlearn library may not be able to train or score all models.
            return None

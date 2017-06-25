import warnings
from functools import lru_cache
from asl_data import SinglesData


def slm_recognize(probabilities: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    sentences = sorted(test_set.sentences_index.items())
    guesses = [word
               for _,sentence in sentences
               for word in best_sentence_interpretation(sentence, probabilities)]
    return guesses


def best_sentence_interpretation(sentence, probabilities):

    @lru_cache
    def _f(sentence, preceding_word=None):
        pass

    return _f(tuple(sentence))

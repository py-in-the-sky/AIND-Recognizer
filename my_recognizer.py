import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
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

    valid_models = {word: model for word,model in models.items() if model is not None}
    probabilities = [word_probabilities(valid_models, *test_set.get_item_Xlengths(i))
                     for i,_ in enumerate(test_set.wordlist)]
    guesses = [best_guess(word_probs) for word_probs in probabilities]
    return probabilities, guesses


def word_probabilities(models, X, lengths):
    word_probs = {}

    for word,model in models.items():
        try:
            word_probs[word] = model.score(X, lengths)
        except ValueError:  # The hmmlearn library may not be able to train or score all models.
            word_probs[word] = float('-inf')

    return word_probs


def best_guess(word_probs):
    return max(word_probs.keys(), key=lambda word: word_probs[word])

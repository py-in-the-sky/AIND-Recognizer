import warnings
from functools import lru_cache
from collections import defaultdict
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



### My SLM ###


def _process_raw_2_gram(raw_2_gram):
    d = defaultdict(dict)

    for likelihood,word1,word2 in raw_2_gram:
        d[word1][word2] = likelihood

    return d


SENTENCE_BEGIN = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_WORD = '[UNKNOWN]'
RAW_1_GRAM =  [(-0.7332026, '</s>'), (-99, '<s>'), (-3.338892, '[UNKNOWN]'), (-2.639822, 'ALL'), (-2.941688, 'ANN'), (-2.241382, 'APPLE'), (-1.764038, 'ARRIVE'), (-2.162145, 'BILL'), (-2.338375, 'BLAME'), (-2.338375, 'BLUE'), (-1.684838, 'BOOK'), (-2.941688, 'BORROW'), (-2.241382, 'BOX'), (-2.338375, 'BOY'), (-2.338375, 'BREAK-DOWN'), (-2.639822, 'BROCCOLI'), (-2.639822, 'BROTHER'), (-2.639822, 'BUT'), (-1.597671, 'BUY'), (-1.709667, 'CAN'), (-2.941688, 'CANDY'), (-1.736002, 'CAR'), (-2.941688, 'CHICAGO'), (-2.639822, 'CHICKEN'), (-2.639822, 'CHINA'), (-2.463453, 'CHOCOLATE'), (-2.639822, 'COAT'), (-2.241382, 'CORN'), (-2.639822, 'DECIDE'), (-2.241382, 'EAT'), (-2.941688, 'FIND'), (-2.241382, 'FINISH'), (-2.639822, 'FISH'), (-2.338375, 'FRANK'), (-2.941688, 'FRED'), (-2.941688, 'FRIEND'), (-1.709667, 'FUTURE'), (-2.941688, 'GET'), (-2.241382, 'GIRL'), (-1.684838, 'GIVE'), (-1.709667, 'GO'), (-2.941688, 'GROUP'), (-2.338375, 'HAVE'), (-2.463453, 'HERE'), (-2.941688, 'HIT'), (-2.941688, 'HOMEWORK'), (-1.940185, 'HOUSE'), (-1.224047, 'IX'), (-2.241382, 'IX-1P'), (-2.639822, 'JANA'), (-0.8869545, 'JOHN'), (-2.463453, 'KNOW'), (-2.941688, 'LAST-WEEK'), (-2.463453, 'LEAVE'), (-2.941688, 'LEG'), (-1.860976, 'LIKE'), (-2.941688, 'LIVE'), (-1.794009, 'LOVE'), (-2.639822, 'MAN'), (-2.941688, 'MANY'), (-1.371861, 'MARY'), (-2.463453, 'MOTHER'), (-2.639822, 'MOVIE'), (-2.941688, 'NAME'), (-2.241382, 'NEW'), (-2.639822, 'NEW-YORK'), (-2.941688, 'NEXT-WEEK'), (-1.940185, 'NOT'), (-2.941688, 'OLD'), (-2.941688, 'PARTY'), (-2.941688, 'PAST'), (-2.338375, 'PEOPLE'), (-1.860976, 'POSS'), (-2.941688, 'POTATO'), (-2.095158, 'PREFER'), (-2.941688, 'PUTASIDE'), (-2.463453, 'READ'), (-2.639822, 'SAY'), (-2.941688, 'SAY-1P'), (-2.941688, 'SEARCH-FOR'), (-1.985961, 'SEE'), (-2.639822, 'SELF'), (-2.463453, 'SELL'), (-2.941688, 'SHOOT'), (-1.860976, 'SHOULD'), (-1.940185, 'SOMETHING-ONE'), (-2.639822, 'STOLEN'), (-2.463453, 'STUDENT'), (-2.338375, 'SUE'), (-2.338375, 'TEACHER'), (-2.241382, 'TELL'), (-2.639822, 'THINK'), (-2.463453, 'THROW'), (-2.941688, 'TOMORROW'), (-2.639822, 'TOY'), (-2.162145, 'VEGETABLE'), (-2.463453, 'VIDEOTAPE'), (-1.898777, 'VISIT'), (-2.639822, 'WANT'), (-1.525109, 'WHAT'), (-1.542145, 'WHO'), (-2.639822, 'WILL'), (-2.095158, 'WOMAN'), (-2.463453, 'WONT'), (-2.941688, 'WRITE'), (-1.940185, 'YESTERDAY')]
RAW_2_GRAM =  [(-1.998848,   '<s>', 'ALL'), (-2.418463,   '<s>', 'ANN'), (-1.938165,   '<s>', 'ARRIVE'), (-1.980628,   '<s>', 'FRANK'), (-2.330257,   '<s>', 'FUTURE'), (-1.292896,   '<s>', 'IX'), (-1.971797,   '<s>', 'IX-1P'), (-0.2383677,  '<s>', 'JOHN'), (-2.418463,   '<s>', 'LAST-WEEK'), (-2.350687,   '<s>', 'LIKE'), (-1.636514,   '<s>', 'LOVE'), (-2.418463,   '<s>', 'MANY'), (-1.429059,   '<s>', 'MARY'), (-2.418463,   '<s>', 'NAME'), (-2.418463,   '<s>', 'NEXT-WEEK'), (-2.372124,   '<s>', 'PEOPLE'), (-1.954655,   '<s>', 'POSS'), (-2.418463,   '<s>', 'SHOOT'), (-1.529894,   '<s>', 'SOMETHING-ONE'), (-2.394676,   '<s>', 'STUDENT'), (-1.783767,   '<s>', 'SUE'), (-1.989642,   '<s>', 'TEACHER'), (-1.783767,   '<s>', 'TELL'), (-1.980628,   '<s>', 'VEGETABLE'), (-1.357784,   '<s>', 'WHAT'), (-1.36421,    '<s>', 'WHO'), (-2.350687,   '<s>', 'WOMAN'), (-0.1017037,  'ALL', 'BOY'), (-0.2328456,  'ANN', 'BLAME'), (-0.4723966,  'APPLE', '</s>'), (-0.2826547,  'APPLE', 'WHO'), (-0.2845587,  'ARRIVE', '</s>'), (-0.7636346,  'ARRIVE', 'HERE'), (-1.397689,   'ARRIVE', 'NOT'), (-0.7551111,  'ARRIVE', 'WHO'), (-0.9710667,  'BILL', 'MARY'), (-1.007225,   'BILL', 'SAY'), (-0.9641827,  'BILL', 'WHAT'), (-0.3614119,  'BILL', 'YESTERDAY'), (-0.3621589,  'BLAME', '</s>'), (-0.8358679,  'BLAME', 'FRED'), (-0.8055056,  'BLAME', 'MARY'), (-0.1731323,  'BLUE', '</s>'), (-0.8349108,  'BLUE', 'SUE'), (-0.05833938, 'BOOK', '</s>'), (-1.48908,    'BOOK', 'PUTASIDE'), (-1.453419,   'BOOK', 'WHAT'), (-0.2347515,  'BORROW', 'VIDEOTAPE'), (-0.03316494, 'BOX', '</s>'), (-0.397842,   'BOY', 'BOOK'), (-0.397842,   'BOY', 'GIVE'), (-0.04186758, 'BREAK-DOWN', '</s>'), (-0.08820622, 'BROCCOLI', '</s>'), (-0.0995997,  'BROTHER', 'ARRIVE'), (-0.1010013,  'BUT', 'CAN'), (-1.390634,   'BUY', '</s>'), (-0.6772217,  'BUY', 'CAR'), (-0.4081662,  'BUY', 'HOUSE'), (-1.107356,   'BUY', 'IX'), (-0.9130853,  'BUY', 'WHAT'), (-1.131315,   'BUY', 'YESTERDAY'), (-0.3952391,  'CAN', '</s>'), (-1.434794,   'CAN', 'BUY'), (-1.028327,   'CAN', 'EAT'), (-1.462365,   'CAN', 'GET'), (-0.4813415,  'CAN', 'GO'), (-0.1990154,  'CANDY', '</s>'), (-1.252332,   'CAR', '</s>'), (-0.6493251,  'CAR', 'BLUE'), (-0.6493251,  'CAR', 'BREAK-DOWN'), (-0.6456196,  'CAR', 'FUTURE'), (-1.418377,   'CAR', 'SHOULD'), (-1.003399,   'CAR', 'STOLEN'), (-0.1990154,  'CHICAGO', '</s>'), (-0.08820622, 'CHICKEN', '</s>'), (-0.09611546, 'CHINA', 'IX'), (-0.2505478,  'CHOCOLATE', '</s>'), (-0.6931846,  'CHOCOLATE', 'WHO'), (-0.08820622, 'COAT', '</s>'), (-0.03316494, 'CORN', '</s>'), (-0.1002999,  'DECIDE', 'VISIT'), (-0.4992962,  'EAT', 'BUT'), (-0.4992962,  'EAT', 'CHICKEN'), (-0.8971162,  'EAT', 'WHAT'), (-0.2318957,  'FIND', 'SOMETHING-ONE'), (-0.4982437,  'FINISH', 'READ'), (-0.9270973,  'FINISH', 'SEE'), (-0.4940593,  'FINISH', 'VISIT'), (-0.1017037,  'FISH', 'WONT'), (-0.7343539,  'FRANK', '</s>'), (-0.8273745,  'FRANK', 'NEW'), (-0.3971492,  'FRANK', 'POSS'), (-0.2197331,  'FRED', 'IX'), (-0.230948,   'FRIEND', 'POSS'), (-0.5407026,  'FUTURE', '</s>'), (-1.428609,   'FUTURE', 'BUY'), (-0.8149922,  'FUTURE', 'FINISH'), (-0.8124353,  'FUTURE', 'GO'), (-1.40304,    'FUTURE', 'JOHN'), (-0.6728719,  'FUTURE', 'NOT'), (-0.2328456,  'GET', 'CAN'), (-0.4989432,  'GIRL', 'BOX'), (-0.2839363,  'GIRL', 'GIVE'), (-1.05176,    'GIVE', 'BOY'), (-1.05176,    'GIVE', 'GIRL'), (-1.037378,   'GIVE', 'GIVE'), (-0.5793824,  'GIVE', 'IX'), (-1.054204,   'GIVE', 'JANA'), (-0.8259363,  'GIVE', 'JOHN'), (-1.05176,    'GIVE', 'TEACHER'), (-0.4810453,  'GO', 'CAN'), (-1.434803,   'GO', 'FUTURE'), (-0.9895954,  'GO', 'IX'), (-1.02938,    'GO', 'MOVIE'), (-1.02938,    'GO', 'NEW-YORK'), (-1.460482,   'GO', 'PARTY'), (-0.8128616,  'GO', 'SHOULD'), (-0.2290586,  'GROUP', 'GIVE'), (-0.397842,   'HAVE', 'BOOK'), (-0.4020332,  'HAVE', 'VIDEOTAPE'), (-0.05677111, 'HERE', '</s>'), (-0.2328456,  'HIT', 'BLAME'), (-0.1990154,  'HOMEWORK', '</s>'), (-0.06123532, 'HOUSE', '</s>'), (-1.232851,   'HOUSE', 'SELL'), (-0.5523194,  'IX', '</s>'), (-1.466943,   'IX', 'BOOK'), (-1.280515,   'IX', 'CAR'), (-1.296308,   'IX', 'GIRL'), (-1.827017,   'IX', 'GIVE'), (-1.901252,   'IX', 'HAVE'), (-1.934905,   'IX', 'HIT'), (-0.8119482,  'IX', 'IX'), (-1.775595,   'IX', 'JOHN'), (-1.150499,   'IX', 'LIKE'), (-1.510776,   'IX', 'MAN'), (-1.885355,   'IX', 'NEW'), (-1.497794,   'IX', 'PEOPLE'), (-1.934905,   'IX', 'SAY-1P'), (-1.485189,   'IX', 'SOMETHING-ONE'), (-1.510776,   'IX', 'THINK'), (-1.787891,   'IX', 'WHO'), (-1.491446,   'IX', 'WOMAN'), (-1.855207,   'IX', 'YESTERDAY'), (-0.9095885,  'IX-1P', 'BUY'), (-0.9318311,  'IX-1P', 'FIND'), (-0.928044,   'IX-1P', 'KNOW'), (-0.4961483,  'IX-1P', 'SEE'), (-0.1020554,  'JANA', 'TOY'), (-1.669684,   'JOHN', '</s>'), (-1.457772,   'JOHN', 'ARRIVE'), (-2.172862,   'JOHN', 'BLAME'), (-1.626682,   'JOHN', 'BOX'), (-2.22746,    'JOHN', 'BROTHER'), (-1.104008,   'JOHN', 'BUY'), (-1.163275,   'JOHN', 'CAN'), (-1.842368,   'JOHN', 'DECIDE'), (-2.199304,   'JOHN', 'FINISH'), (-1.842368,   'JOHN', 'FISH'), (-1.061685,   'JOHN', 'FUTURE'), (-1.279418,   'JOHN', 'GIVE'), (-1.371495,   'JOHN', 'GO'), (-1.184636,   'JOHN', 'IX'), (-2.172862,   'JOHN', 'IX-1P'), (-1.634037,   'JOHN', 'LEAVE'), (-1.292735,   'JOHN', 'LIKE'), (-1.292735,   'JOHN', 'LOVE'), (-1.727236,   'JOHN', 'MARY'), (-2.199304,   'JOHN', 'MOTHER'), (-1.807824,   'JOHN', 'NOT'), (-2.257569,   'JOHN', 'PAST'), (-1.467774,   'JOHN', 'POSS'), (-1.807824,   'JOHN', 'PREFER'), (-2.22746,    'JOHN', 'READ'), (-2.22746,    'JOHN', 'SAY'), (-2.257569,   'JOHN', 'SEARCH-FOR'), (-1.296129,   'JOHN', 'SEE'), (-1.292735,   'JOHN', 'SHOULD'), (-1.786232,   'JOHN', 'VISIT'), (-1.842368,   'JOHN', 'WANT'), (-1.736529,   'JOHN', 'WHO'), (-1.842368,   'JOHN', 'WILL'), (-2.257569,   'JOHN', 'WRITE'), (-2.124368,   'JOHN', 'YESTERDAY'), (-0.06165877, 'KNOW', 'IX'), (-0.2253044,  'LAST-WEEK', 'JOHN'), (-0.06165877, 'LEAVE', 'IX'), (-0.1990154,  'LEG', '</s>'), (-1.153652,   'LIKE', '</s>'), (-0.6665106,  'LIKE', 'CHOCOLATE'), (-0.6654365,  'LIKE', 'CORN'), (-0.5115523,  'LIKE', 'IX'), (-1.261816,   'LIKE', 'MARY'), (-0.2357076,  'LIVE', 'CHICAGO'), (-0.6844902,  'LOVE', '</s>'), (-1.2899, 'LOVE', 'IX'), (-0.9228759,  'LOVE', 'JOHN'), (-0.5812182,  'LOVE', 'MARY'), (-0.7167907,  'LOVE', 'WHAT'), (-1.323714,   'LOVE', 'WHO'), (-0.5044708,  'MAN', 'IX'), (-0.5300937,  'MAN', 'NEW'), (-0.2337975,  'MANY', 'PEOPLE'), (-0.223712,   'MARY', '</s>'), (-1.786139,   'MARY', 'BILL'), (-1.768141,   'MARY', 'BLAME'), (-1.768141,   'MARY', 'IX-1P'), (-1.329274,   'MARY', 'JOHN'), (-1.148073,   'MARY', 'LOVE'), (-1.366089,   'MARY', 'SELF'), (-1.150193,   'MARY', 'VEGETABLE'), (-1.750859,   'MARY', 'VISIT'), (-1.786139,   'MARY', 'WONT'), (-0.2729032,  'MOTHER', 'ARRIVE'), (-0.680562,   'MOTHER', 'IX'), (-0.4654046,  'MOVIE', '</s>'), (-0.5357867,  'MOVIE', 'TOMORROW'), (-0.1990154,  'NAME', '</s>'), (-0.2847928,  'NEW', 'CAR'), (-0.4996457,  'NEW', 'COAT'), (-0.08820622, 'NEW-YORK', '</s>'), (-0.2253044,  'NEXT-WEEK', 'JOHN'), (-1.074471,   'NOT', '</s>'), (-0.3349186,  'NOT', 'BUY'), (-1.217912,   'NOT', 'LIKE'), (-0.7909485,  'NOT', 'VISIT'), (-1.174245,   'NOT', 'WHAT'), (-0.2347515,  'OLD', 'HOUSE'), (-0.2318957,  'PARTY', 'FUTURE'), (-0.2357076,  'PAST', 'LIVE'), (-0.8163024,  'PEOPLE', 'GIVE'), (-0.8358679,  'PEOPLE', 'GROUP'), (-0.3992364,  'PEOPLE', 'PREFER'), (-0.8559346,  'POSS', 'BOOK'), (-1.298955,   'POSS', 'BROTHER'), (-1.307339,   'POSS', 'CANDY'), (-0.8619646,  'POSS', 'CAR'), (-1.290729,   'POSS', 'FRANK'), (-1.307339,   'POSS', 'FRIEND'), (-1.307339,   'POSS', 'LEG'), (-0.8680796,  'POSS', 'NEW'), (-1.307339,   'POSS', 'OLD'), (-0.2234393,  'POTATO', 'WHAT'), (-0.645075,   'PREFER', 'BROCCOLI'), (-0.6436735,  'PREFER', 'CORN'), (-0.6394957,  'PREFER', 'GO'), (-1.077959,   'PREFER', 'POTATO'), (-0.1990154,  'PUTASIDE', '</s>'), (-0.06380112, 'READ', 'BOOK'), (-0.5152675,  'SAY', 'JOHN'), (-0.5152675,  'SAY', 'MARY'), (-0.2328456,  'SAY-1P', 'LOVE'), (-0.2262399,  'SEARCH-FOR', 'WHO'), (-1.098014,   'SEE', 'IX'), (-1.126834,   'SEE', 'JOHN'), (-1.126834,   'SEE', 'MARY'), (-0.541358,   'SEE', 'THROW'), (-1.117013,   'SEE', 'WHAT'), (-0.7329912,  'SEE', 'WHO'), (-0.1010013,  'SELF', 'PREFER'), (-0.2742958,  'SELL', 'CAR'), (-0.7024307,  'SELL', 'YESTERDAY'), (-0.2337975,  'SHOOT', 'FRANK'), (-0.4886818,  'SHOULD', '</s>'), (-1.277342,   'SHOULD', 'BUY'), (-1.29894,    'SHOULD', 'FINISH'), (-1.288006,   'SHOULD', 'GO'), (-0.6624442,  'SHOULD', 'NOT'), (-0.8721989,  'SHOULD', 'SHOULD'), (-1.186045,   'SOMETHING-ONE', 'ARRIVE'), (-1.230033,   'SOMETHING-ONE', 'BORROW'), (-1.198165,   'SOMETHING-ONE', 'CAR'), (-0.7868468,  'SOMETHING-ONE', 'POSS'), (-0.7964872,  'SOMETHING-ONE', 'STUDENT'), (-1.151592,   'SOMETHING-ONE', 'WHAT'), (-0.7916402,  'SOMETHING-ONE', 'WOMAN'), (-0.08820622, 'STOLEN', '</s>'), (-0.2763931,  'STUDENT', 'HAVE'), (-0.7043038,  'STUDENT', 'SOMETHING-ONE'), (-0.04710582, 'SUE', 'BUY'), (-0.4020332,  'TEACHER', 'APPLE'), (-0.397842,   'TEACHER', 'GIVE'), (-0.03787356, 'TELL', 'BILL'), (-0.09820265, 'THINK', 'MARY'), (-0.06509162, 'THROW', 'APPLE'), (-0.1990154,  'TOMORROW', '</s>'), (-0.08820622, 'TOY', '</s>'), (-0.5781283,  'VEGETABLE', 'CHINA'), (-0.9997487,  'VEGETABLE', 'IX-1P'), (-0.5767267,  'VEGETABLE', 'KNOW'), (-0.9997487,  'VEGETABLE', 'PREFER'), (-0.05677111, 'VIDEOTAPE', '</s>'), (-0.1582235,  'VISIT', 'MARY'), (-1.266712,   'VISIT', 'MOTHER'), (-1.227424,   'VISIT', 'WHAT'), (-1.237792,   'VISIT', 'WHO'), (-0.1017037,  'WANT', 'SELL'), (-0.4607335,  'WHAT', '</s>'), (-1.594247,   'WHAT', 'ARRIVE'), (-0.9897979,  'WHAT', 'BOOK'), (-0.8464446,  'WHAT', 'JOHN'), (-0.8464446,  'WHAT', 'MARY'), (-1.614903,   'WHAT', 'SOMETHING-ONE'), (-1.20524,    'WHAT', 'WOMAN'), (-1.199784,   'WHAT', 'YESTERDAY'), (-0.5463784,  'WHO', '</s>'), (-1.563978,   'WHO', 'ARRIVE'), (-1.48926,    'WHO', 'IX'), (-0.9612188,  'WHO', 'JOHN'), (-0.6395192,  'WHO', 'MARY'), (-1.606785,   'WHO', 'MOTHER'), (-1.580598,   'WHO', 'POSS'), (-1.192349,   'WHO', 'TELL'), (-1.606785,   'WHO', 'VEGETABLE'), (-1.152642,   'WHO', 'WHAT'), (-0.1002999,  'WILL', 'VISIT'), (-0.6326226,  'WOMAN', 'ARRIVE'), (-0.6326226,  'WOMAN', 'BOOK'), (-1.067628,   'WOMAN', 'HAVE'), (-1.003036,   'WOMAN', 'IX'), (-1.019343,   'WOMAN', 'WHAT'), (-0.06509162, 'WONT', 'EAT'), (-0.2357076,  'WRITE', 'HOMEWORK'), (-0.7346112,  'YESTERDAY', '</s>'), (-0.7875246,  'YESTERDAY', 'BOOK'), (-1.157938,   'YESTERDAY', 'IX'), (-0.777412,   'YESTERDAY', 'WHAT'), (-0.5767061,  'YESTERDAY', 'WHO')]
# Raw n-gram data from the link provided in the notebook:
# ftp://wasserstoff.informatik.rwth-aachen.de/pub/rwth-boston-104/lm/
PROCESSED_1_GRAM = {word: likelihood for likelihood,word in RAW_1_GRAM}
PROCESSED_2_GRAM = _process_raw_2_gram(RAW_2_GRAM)
NOT_FOUND_LIKELIHOOD = -100


def slm_recognize(probabilities: dict, test_set: SinglesData):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    @lru_cache
    def _best_sentence_interpretation(sentence, preceding_word=SENTENCE_BEGIN):
        # NOTE: all probability and LM data are log likelihoods, not actual probabilities,
        # so use addition to combine them, not multiplication.
        # NOTE: I use the 'L' prefix to indicate likelihoods. E.g., Lbigram means
        # "the likelihood of this bigram."

        if not sentence:
            interpretation = (SENTENCE_END,)
            Lword = PROCESSED_1_GRAM.get(SENTENCE_END, NOT_FOUND_LIKELIHOOD)
            Lbigram = PROCESSED_2_GRAM.get(preceding_word, {}).get(SENTENCE_END, NOT_FOUND_LIKELIHOOD)
            Linterpretation = Lbigram + Lword
            return interpretation, Linterpretation

        best_interpretation = None
        Linterpretation = float('-inf')
        current_word_id, suffix = sentence[0], sentence[1:]

        for word,Lword_given_video in probabilities[current_word_id].items():
            Lword = PROCESSED_1_GRAM.get(word, NOT_FOUND_LIKELIHOOD)
            Lbigram = PROCESSED_2_GRAM.get(preceding_word, {}).get(word, NOT_FOUND_LIKELIHOOD)
            suffix_interpretation, Lsuffix = _best_sentence_interpretation(suffix, preceding_word=word)
            Loverall = Lword + Lbigram + Lword_given_video + Lsuffix

            if Loverall > Linterpretation:
                Linterpretation = Loverall
                best_interpretation = (word,) + suffix_interpretation

        return best_interpretation, Linterpretation

    sentences = sorted(test_set.sentences_index.items())  # Sort so that word guesses align with word_ids.
    sentence_guesses = (_best_sentence_interpretation(tuple(sentence))[0]
                        for _,sentence in sentences)
    word_guesses = [word for sentence in sentence_guesses for word in sentence]
    return word_guesses

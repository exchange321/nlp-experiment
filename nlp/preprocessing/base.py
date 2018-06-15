import re

import nltk
import RAKE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from autocorrect import spell

TRANSLATION = {
    'ca': ['can'],
    'Ca': ['Can'],
    "n't": ['not'],
    "'m": ['am'],
    "'s": ['is'],
    "'ve": ['have'],
    'u': ['you'],
    'ha': ['have'],
    'wo': ['will'],
    'atm': ['at', 'the', 'moment'],
    'xmas': ['Christmas'],
    "'ll": ['will'],
    'im': ['I', 'am']
}

STOPWORDS = stopwords.words('english') + []

GRAMMER = r'''
    NBAR:
        {<NN.*|JJ>*<NN.*>}

    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}

    PP:
        {<IN><NP>}

    VP:
        {<VB.*><NP|PP|CLAUSE>+$}

    CLAUSE:
        {<NP><VP>}
'''

WORDNET_LEMMATIZER = WordNetLemmatizer()
PORTER_STEMMER = PorterStemmer()

RAKE = RAKE.Rake(STOPWORDS)
CHUNKER = nltk.RegexpParser(GRAMMER)


def sentensize(text):
    return nltk.tokenize.sent_tokenize(text)


def tokenize(text):
    return nltk.tokenize.word_tokenize(text)


def translate(raw_tokens):
    tokens = []
    for raw_token in raw_tokens:
        if raw_token in TRANSLATION:
            tokens += TRANSLATION[raw_token]
        else:
            tokens.append(raw_token)
    return tokens


def normalize(tokens):
    return [token.lower()
            for token in tokens]


def sanitize(tokens):
    return [token for token in tokens
            if not re.search(r'^\W+$', token)]


def removeStopWords(tokens):
    return [token
            for token in tokens
            if token not in STOPWORDS]


def lemmatize(tokens):
    return [WORDNET_LEMMATIZER.lemmatize(token) for token in tokens]


def stem(tokens):
    return [PORTER_STEMMER.stem(token) for token in tokens]


def phrases(tokens):
    sentence = ' '.join(tokens)
    rakePhrases = _getRakePhrases(sentence)
    posPhrases = _getPosPhrases(tokens)

    phrases = list(set(rakePhrases + posPhrases))

    return phrases


def _getRakePhrases(sentence):
    return [_normalizePhrase(phrase) for phrase, _ in RAKE.run(
        sentence) if len(phrase.split(' ')) > 1]


def _getPosPhrases(tokens):
    posTokens = nltk.tag.pos_tag(tokens)
    tree = CHUNKER.parse(posTokens)
    return [_normalizePhrase(' '.join(term))
            for term in _getTerms(tree) if len(term) > 1]


def _normalizePhrase(phrase):
    tokens = phrase.split(' ')
    tokens = normalize(tokens)
    tokens = sanitize(tokens)
    tokens = correct(tokens)

    tokens = removeStopWords(tokens)
    tokens = lemmatize(tokens)

    return ' '.join(tokens)


def _getTerms(tree):
    for leaf in _getLeaves(tree):
        term = [w for w, _ in leaf]
        yield term


def _getLeaves(tree):
    for subtree in tree.subtrees(
        filter=lambda t: t.label() in ['NP', 'VP']
    ):
        yield subtree.leaves()


def correct(tokens):
    return [spell(token)
            if not re.search(r'^\W+$', token)
            else token
            for token in tokens]


def flatten(x):
    return [a for i in x for a in i]


def replaceWithPhrases(tokens, phrases):
    tempTokens = tokens.copy()
    for phrase in phrases:
        phraseTokens = phrase.split(' ')
        isPhrase = False
        for i in range(len(tempTokens) + 1 - len(phraseTokens)):
            matches = 0
            for key, token in enumerate(phraseTokens):
                if tempTokens[i + key] == token:
                    matches += 1
            if matches == len(phraseTokens):
                isPhrase = True
                break

        if isPhrase:
            start = tempTokens.index(phraseTokens[0])
            end = start + len(phraseTokens)
            tempTokens[start:end] = [' '.join(phraseTokens)]
    return tempTokens


def appendWithPhrases(tokens, phrases):
    return tokens + phrases


def nGram(tokens, maxN=3):
    output = tokens[:]
    for n in range(1, maxN + 1):
        output += _nGram(tokens, n)

    return output


def _nGram(tokens, n=2):
    output = []
    for i in range(len(tokens) - n + 1):
        output.append(' '.join(tokens[i:i + n]))
    return output

import operator
from collections import Counter

import nlp.preprocessing.base as preprocessing


class DF:
    def __init__(self, paragraphs):
        self.paragraphs = paragraphs.copy()

    def process(self):
        self._preprocessing()
        self._calDf()

    def _preprocessing(self):
        self.splittedParagraphs = self._tokenizeToSentences()
        self.tokenizedParagraphs = self._tokenizeToTokens()
        self.flattenedParagraphs = self._flattenToParagraph()
        self.bagOfTermsInParagraphs = self._calBagOfTerms()
        self.documentFrequency = self._calDocumentFrequency()
        self.documentFrequency = self._filterExtreme(
            self.documentFrequency)

    def _calDf(self):
        self.df = self._calculateDf()
        return self.df

    def _tokenizeToSentences(self):
        return [preprocessing.sentensize(paragraph)
                for paragraph in self.paragraphs]

    def _tokenizeToTokens(self):
        tokenizedParagraphs = []
        for key, paragraph in enumerate(self.splittedParagraphs):
            tokenizedParagraphs.append([])
            for sentence in paragraph:
                tokens = preprocessing.tokenize(sentence)
                tokens = preprocessing.translate(tokens)

                phrases = preprocessing.phrases(tokens)

                tokens = preprocessing.normalize(tokens)
                tokens = preprocessing.sanitize(tokens)
                tokens = preprocessing.correct(tokens)
                tokens = preprocessing.removeStopWords(tokens)
                tokens = preprocessing.lemmatize(tokens)

                tokens = preprocessing.replaceWithPhrases(tokens, phrases)

                tokenizedParagraphs[key].append(tokens)
        return tokenizedParagraphs

    def _flattenToParagraph(self):
        return [preprocessing.flatten(tokenizedParagraph)
                for tokenizedParagraph in self.tokenizedParagraphs]

    def _calBagOfTerms(self):
        return [Counter(flattenedParagraph)
                for flattenedParagraph in self.flattenedParagraphs]

    def _calDocumentFrequency(self):
        return Counter([term
                        for paragraph in self.bagOfTermsInParagraphs
                        for term in paragraph.keys()])

    def _filterExtreme(self, df, noBelow=1, noAbove=0.5):
        n = sum(df.values())
        noAbove = n * noAbove
        copy = df.copy()
        for term in copy:
            if df[term] < noBelow or df[term] >= noAbove:
                del df[term]
        return df

    def _calculateDf(self):
        df = self.documentFrequency.copy()
        return sorted(df.items(), key=operator.itemgetter(1), reverse=True)

    def __str__(self):
        output = 'Top keyphrases:\n'
        if len(self.df) > 30:
            df = self.df[:30]
        else:
            df = self.df[:]
        output += self._formatString(df)
        return output

    def _formatString(self, df):
        output = ''
        for key, (keyphrase, score) in enumerate(df):
            output += f'\t{(key + 1):2}: {keyphrase:55} - {score:10.3f}\n'
        return output

    @staticmethod
    def PROCESS(paragraphs):
        Df = DF(paragraphs.copy())
        Df.process()

        return Df

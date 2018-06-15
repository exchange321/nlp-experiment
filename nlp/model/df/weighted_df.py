import os
import statistics

import numpy as np
import requests
from dotenv import load_dotenv, find_dotenv
import nlp.preprocessing.base as preprocessing

from nlp.model.df.base import DF

load_dotenv(find_dotenv())

ANALYTICS_URL = os.getenv('ANALYTICS_URL')
AUTH_TOKEN = os.getenv('AUTH_TOKEN')


class WeightedDF(DF):
    def __init__(self, paragraphs, ratings, scale):
        super().__init__(paragraphs.copy())
        self.ratings = ratings.copy()
        self.scale = scale

    def _preprocessing(self):
        self.normalizedRatings = self._normalizeScores(
            self.ratings, self.scale, (1, 2))
        self.splittedParagraphs = self._tokenizeToSentences()
        self.sentimentScores = [self._normalizeScores(
            np.array(scores, dtype=np.float),
            (-1, 1), (1, 2)
        ).tolist() for scores in self._getSentimentScores()]
        self.meanSentimentScores = self._calMeanScores(self.sentimentScores)
        self.meanScores = self._calMeanScores(
            np.concatenate(
                (
                    np.array(self.normalizedRatings)[np.newaxis],
                    np.array(self.meanSentimentScores)[np.newaxis]
                )
            ).T
        )
        self.adjustedSentimentScores = [self._normalizeScores(
            np.array(scores, dtype=np.float),
            (1, 2), (-1, 1)
        ).tolist() for scores in self._adjustSentimentScores()]
        self.scoredSentences = self._scoreSentences()
        (self.positiveSentences, self.neutralSentences,
         self.negativeSentences) = self._categorizeSentences((-0.2, 0.2))
        self.categorizedSentences = [
            self._transposeSentences(self.positiveSentences),
            self._transposeSentences(self.neutralSentences),
            self._transposeSentences(self.negativeSentences)]
        for key, sentences in enumerate(self.categorizedSentences.copy()):
            tokenizedSentences = self._tokenizeToTokens(sentences[1])
            self.categorizedSentences[key][1] = tokenizedSentences
        self.bagOfTermsInCategories = [self._calBagOfTerms(
            sentences) for sentences in self.categorizedSentences]

    def _normalizeScores(self, scores, srcScale=(1, 10), destScale=(-1, 1)):
        (min, max) = srcScale
        (a, b) = destScale

        normalizedScores = np.fromiter(
            ((((b - a) * (x - min)) / (max - min)) + a for x in scores),
            scores.dtype
        )
        return normalizedScores

    def _getSentimentScores(self):
        headers = {
            'Authorization': 'Bearer ' + AUTH_TOKEN,
            'Content-Type': 'application/json'
        }
        output = []
        for paragraph in self.splittedParagraphs:
            data = {
                'texts': paragraph
            }

            rawResponse = requests.post(
                ANALYTICS_URL + 'detect_sentiment',
                json=data,
                headers=headers,
                verify=False
            )
            jsonResponse = rawResponse.json()
            results = jsonResponse['results']
            scores = np.empty([len(results)])
            for result in results:
                scores[result['index']] = result['score']
            output.append(scores.tolist())
        return output

    def _adjustSentimentScores(self):
        output = []
        for key, score in enumerate(self.sentimentScores):
            adjustedScore = []
            for item in score:
                if item == 0 and self.meanSentimentScores[key] == 0:
                    percentage = 1
                else:
                    percentage = item / self.meanSentimentScores[key]
                adjustedScore.append(self.meanScores[key] * percentage)
            output.append(adjustedScore)
        return output

    def _scoreSentences(self):
        output = []
        for i, paragraph in enumerate(self.splittedParagraphs):
            sentences = []
            for j, sentence in enumerate(paragraph):
                sentences.append(
                    (self.adjustedSentimentScores[i][j], sentence))
            output.append(sentences)
        return output

    def _categorizeSentences(self, breakpoint=(0, 0)):
        positive = []
        neutral = []
        negative = []
        for sentences in self.scoredSentences:
            for sentence in sentences:
                if sentence[0] < breakpoint[0]:
                    negative.append((breakpoint[0] - sentence[0], sentence[1]))
                elif sentence[0] > breakpoint[1]:
                    positive.append((sentence[0] + breakpoint[1], sentence[1]))
                else:
                    neutral.append((sentence[0] + breakpoint[1], sentence[1]))
        return (positive, neutral, negative)

    def _transposeSentences(self, sentences):
        scores = []
        rawSentences = []
        for score, sentence in sentences:
            scores.append(score)
            rawSentences.append(sentence)
        return [scores, rawSentences]

    def _tokenizeToTokens(self, sentences):
        tokenizedSentences = []
        for sentence in sentences:
            tokens = preprocessing.tokenize(sentence)
            tokens = preprocessing.translate(tokens)

            phrases = preprocessing.phrases(tokens)

            tokens = preprocessing.normalize(tokens)
            tokens = preprocessing.sanitize(tokens)
            tokens = preprocessing.correct(tokens)

            tokens = preprocessing.removeStopWords(tokens)
            tokens = preprocessing.lemmatize(tokens)

            tokens = preprocessing.replaceWithPhrases(tokens, phrases)

            tokenizedSentences.append(tokens)
        return tokenizedSentences

    def _calBagOfTerms(self, sentences):
        filteredSentences = [list(set(sentence)) for sentence in sentences[1]]
        scoredTokens = {}
        for key, sentence in enumerate(filteredSentences):
            for token in sentence:
                if token not in scoredTokens:
                    scoredTokens[token] = []
                scoredTokens[token].append(sentences[0][key])
        for token in scoredTokens:
            scores = scoredTokens[token]
            scoredTokens[token] = sum(scores)
        return scoredTokens

    def _combineLists(self, *lists):
        output = []
        for key, item in enumerate(lists[0]):
            output.append(tuple([listitem[key] for listitem in lists]))
        return output

    def _calMeanScores(self, scores):
        return [statistics.mean(score) for score in scores]

    def _calculateDf(self):
        df = []
        for category in self.bagOfTermsInCategories.copy():
            result = sorted(category.items(), key=lambda x: x[1], reverse=True)
            df.append(result)
        return df

    def _calDf(self):
        self.df = self._calculateDf()
        return self.df

    def __str__(self):
        titles = ('Positive', 'Neutral', 'Negative')
        output = ''
        for key, title in enumerate(titles):
            output += f'{title} - Top Keyphrases: \n'
            df = self.df[key]
            if len(df) > 15:
                df = df[:15]
            output += self._formatString(df)
            output += '\n'
        return output

    @staticmethod
    def PROCESS(paragraphs, ratings, scale):
        WeightedDf = WeightedDF(
            paragraphs,
            ratings,
            scale
        )
        WeightedDf.process()

        return WeightedDf

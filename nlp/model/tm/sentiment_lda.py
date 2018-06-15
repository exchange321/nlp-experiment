import copy
import itertools
import os
import pprint
import statistics
import uuid

import numpy as np
import requests
from dotenv import find_dotenv, load_dotenv
from gensim import corpora, models

import nlp.preprocessing.base as preprocessing
from nlp.model.tm.lda import LDA

load_dotenv(find_dotenv())

ANALYTICS_URL = os.getenv('ANALYTICS_URL')
AUTH_TOKEN = os.getenv('AUTH_TOKEN')

pp = pprint.PrettyPrinter(indent=4)


class SentimentLDA(LDA):
    def __init__(self, paragraphs, ratings, scale):
        super().__init__(paragraphs.copy())
        self.ratings = ratings.copy()
        self.scale = scale

    def preprocessing(self, filter=3):
        self.normalizedRatings = self._normalizeScores(
            self.ratings, self.scale, (1, 2)
        )
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

        self.collections = [
            self._transposeSentences(self.positiveSentences),
            self._transposeSentences(self.neutralSentences),
            self._transposeSentences(self.negativeSentences)]

        for key, sentences in enumerate(self.collections.copy()):
            tokenizedSentences = self._tokenizeToTokens(sentences[1])
            self.collections[key] = [sentence for sentence
                                     in tokenizedSentences if sentence]

        [positives, neutral, negatives] = self.collections
        self.collections = [
            {
                'collection': positives
            },
            {
                'collection': neutral
            },
            {
                'collection': negatives
            }
        ]
        self._prepare(filter)

    def _prepare(self, filter=None):
        for key, collection in enumerate(self.collections.copy()):
            dictionary = corpora.Dictionary(
                document
                for document in collection['collection'])

            if filter is not None:
                dictionary.filter_extremes(filter)

            corpus = [dictionary.doc2bow(document)
                      for document
                      in collection['collection']]

            self.collections[key]['dictionary'] = dictionary
            self.collections[key]['corpus'] = corpus

    def _train(self, num_topics, collection):
        return models.ldamulticore.LdaMulticore(
            collection['corpus'],
            workers=2,
            num_topics=num_topics,
            id2word=collection['dictionary'],
            passes=20,
            iterations=400
        )

    def train(self, num_topics=None):
        if num_topics is not None:
            self.ldamodels = [self._train(num_topics[key], collection)
                              for key, collection
                              in enumerate(self.collections)]

        else:
            self.ldamodels = []
            for collection in self.collections:
                highest = {
                    'num_topic': 0,
                    'coherence': 0
                }
                for num in itertools.count(1):
                    ldamodel = self._train(num, collection)
                    cm = models.CoherenceModel(
                        model=ldamodel,
                        texts=collection['collection'],
                        dictionary=collection['dictionary'],
                        coherence='c_v'
                    )
                    coherence = cm.get_coherence()
                    if coherence > highest['coherence']:
                        highest = {
                            'lda': copy.deepcopy(ldamodel),
                            'num_topic': num,
                            'coherence': coherence
                        }
                    elif ((highest['coherence'] - coherence) > 0.2) \
                            or num >= 20:
                        break

                self.ldamodels.append(highest['lda'])

    def getTopics(self, ldamodel):
        return ldamodel.show_topics(
            num_topics=np.inf,
            num_words=10,
            formatted=False
        )

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

    def _calMeanScores(self, scores):
        return [statistics.mean(score) for score in scores]

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

    def json(self, name):
        titles = ('positive', 'neutral', 'negative')
        output = []
        for key, title in enumerate(titles):
            collection = {
                'name': name,
                'class': title,
            }
            ldamodel = self.ldamodels[key]
            topics = self.getTopics(ldamodel)
            children = []

            for _, topic in topics:
                children.append({
                    'name': str(uuid.uuid4()),
                    'children': [
                        {
                            'name': keyphrase,
                            'size': score.item()
                        }
                        for keyphrase, score in topic
                    ]
                })

            collection['children'] = children
            output.append(collection)

        return output

    def __str__(self):
        titles = ('Positive', 'Neutral', 'Negative')
        output = ''
        for key, title in enumerate(titles):
            ldamodel = self.ldamodels[key]
            topics = self.getTopics(ldamodel)
            output += f"#####{f'{title} - {len(topics):02d} Topics ':^70}#####"
            output += '\n\n'
            for i, topic in topics:
                output += f'Topic {i+1:02d}:\n'
                output += self._formatString(topic)
                output += '\n'
        return output

    @staticmethod
    def PROCESS(paragraphs, ratings, scale, num_topics=None):
        sentimentLda = SentimentLDA(paragraphs, ratings, scale)
        sentimentLda.preprocessing(None)
        sentimentLda.train(num_topics)

        return sentimentLda

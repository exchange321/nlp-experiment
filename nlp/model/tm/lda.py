import copy
import itertools
import uuid

import numpy as np
from gensim import corpora, models

import nlp.preprocessing.base as preprocessing


class LDA:
    def __init__(self, paragraphs):
        self.paragraphs = paragraphs

    def preprocessing(self, filter=3):
        self.splittedParagraphs = self._tokenizeToSentences()
        self.tokenizedParagraphs = self._tokenizeToTokens()
        self.collection = self._flattenToParagraph()

        self._prepare(filter)

    def _train(self, num_topics):
        self.ldamodel = models.ldamulticore.LdaMulticore(
            self.corpus,
            workers=2,
            num_topics=num_topics,
            id2word=self.dictionary,
            passes=20,
            iterations=400
        )

    def train(self, num_topics=None):
        if num_topics is not None:
            self._train(num_topics)

        else:
            highest = {
                'num_topic': 0,
                'coherence': 0
            }
            for num in itertools.count(1):
                self._train(num)
                cm = models.CoherenceModel(
                    model=self.ldamodel,
                    texts=self.collection,
                    dictionary=self.dictionary,
                    coherence='c_v'
                )
                coherence = cm.get_coherence()
                if coherence > highest['coherence']:
                    highest = {
                        'lda': copy.deepcopy(self.ldamodel),
                        'num_topic': num,
                        'coherence': coherence
                    }
                elif ((highest['coherence'] - coherence) > 0.2) \
                        or num >= 20:
                    break

            self.ldamodel = highest['lda']

    def getTopics(self):
        return self.ldamodel.show_topics(
            num_topics=np.inf,
            num_words=10,
            formatted=False
        )

    def _prepare(self, filter=None):
        self.dictionary = corpora.Dictionary(
            document for document in self.collection)

        if filter is not None:
            self.dictionary.filter_extremes(filter)

        self.corpus = [self.dictionary.doc2bow(document)
                       for document in self.collection]

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

    def json(self, name):
        output = []
        collection = {
            'name': name,
            'class': 'nosentiment',
        }
        children = []

        for _, topic in self.getTopics().copy():
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
        output = ''

        topics = self.getTopics()
        for i, topic in topics:
            output += f'Topic {i+1:02d}:\n'
            output += self._formatString(topic)
            output += '\n'

        return output

    def _formatString(self, token):
        output = ''
        for key, (keyphrase, score) in enumerate(token):
            output += f'\t{(key + 1):2}: {keyphrase:55} - {score:10.3f}\n'
        return output

    @staticmethod
    def PROCESS(paragraphs, num_topics=None):
        lda = LDA(paragraphs)
        lda.preprocessing()
        lda.train(num_topics)

        return lda

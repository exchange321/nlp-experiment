import pprint
import warnings

# from nlp.model.df.base import DF
# from nlp.model.df.weighted_df import WeightedDF
from nlp.visualization.console.base import Results, Result
from helpers.database import Database
from helpers.output import output
from nlp.model.tm.lda import LDA
from nlp.model.tm.sentiment_lda import SentimentLDA

warnings.simplefilter("ignore")

pp = pprint.PrettyPrinter(indent=4)


tenants = [
    'master_large_demo',
    'vision6',
    'fujitsu',
    'f5mgroup',
    'sheldon',
    'plantminer',
    'q4financial'
]

results = Results()

for tenant in tenants:
    database = Database(tenant)
    tenantName = database.getName()

    print(f'Processing {tenantName}...')

    database.process()
    ratingTexts = database.getRatingTexts()

    for ratingText in ratingTexts:
        (ratings, paragraphs) = database.getData(ratingText)

        if len(paragraphs) < 5:
            continue

        result = Result(
            tenantName,
            ratingText['formDesignName'],
            paragraphs
        )

        lda = LDA.PROCESS(paragraphs)
        result.add(f'LDA - {len(lda.getTopics()):2} Topics', lda)

        result.add('Sentiment LDA Result', SentimentLDA.PROCESS(
            paragraphs,
            ratings,
            ratingText['rating']['scale']
        ))

        results.add(result)


output(results, 'results.txt')
output(results, 'results.json', True)

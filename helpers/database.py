from nlp.database.base import DatabaseCursor
import numpy as np


class Database:
    def __init__(self, tenant):
        self.tenant = tenant

    def process(self):
        self._getHappinessRating()
        self._getHappinessText()

    def getName(self):
        with DatabaseCursor(self.tenant) as cur:
            cur.execute(
                """
                SELECT value
                FROM configuration
                WHERE key = 'tenant_name'
                """
            )
            (tenantName,) = cur.fetchone()
            self.tenantName = tenantName

            return self.tenantName

    def _getHappinessRating(self):
        with DatabaseCursor(self.tenant) as cur:
            cur.execute(
                """
                SELECT id, definition
                FROM metrics
                WHERE name = 'Happiness Rating' and is_system
                """
            )
            (happinessRatingId, happinessRatingDef) = cur.fetchone()
            self.happinessRatingId = happinessRatingId
            self.happinessRatingDef = happinessRatingDef

            return (self.happinessRatingId, self.happinessRatingDef)

    def _getHappinessText(self):
        with DatabaseCursor(self.tenant) as cur:
            cur.execute(
                """
                SELECT id, definition
                FROM metrics
                WHERE name = 'Happiness Text' and is_system
                """
            )
            (happinessTextId, happinessTextDef) = cur.fetchone()
            self.happinessTextId = happinessTextId
            self.happinessTextDef = happinessTextDef

            return (self.happinessTextId, self.happinessTextDef)

    def getRatingTexts(self):
        ratingTexts = [
            {
                'formDesignId': formDesign['form_design_id'],
                'rating': {
                    'formFieldId': formDesign['form_field_id']
                },
                'texts': [
                    {
                        'formFieldId': textFormDesign['form_field_id']
                    } for textFormDesign in self.happinessTextDef
                    if textFormDesign['form_design_id']
                    == formDesign['form_design_id']
                ]
            } for formDesign in self.happinessRatingDef
        ]
        self.ratingTexts = ratingTexts

        return self.ratingTexts

    def _getAnswers(self, ratingText):
        with DatabaseCursor(self.tenant) as cur:
            cur.execute(
                f"""
                SELECT name, fields
                FROM form_designs
                WHERE id = '{ratingText['formDesignId']}'
                """
            )

            (formDesignName, ratingFields,) = cur.fetchone()
            ratingText['formDesignName'] = formDesignName

            field = ratingFields[ratingText['rating']['formFieldId']]
            if 'options' in field:
                points = [float(point['value']) for point in field['options']]
                ratingText['rating']['scale'] = (min(points), max(points))
            elif 'start' in field and 'end' in field:
                ratingText['rating']['scale'] = (
                    float(field['start']),
                    float(field['end'])
                )

            cur.execute(
                f"""
                SELECT answers
                FROM form_instances
                WHERE form_design_id = '{ratingText['formDesignId']}'
                    AND answers IS NOT NULL
                """
            )

            answers = [answer for (answer,) in cur.fetchall()]

            return answers

    def getData(self, ratingText):
        answers = self._getAnswers(ratingText)

        ratings = []
        paragraphs = []
        for rawAnswer in answers.copy():
            if ratingText['rating']['formFieldId'] not in rawAnswer:
                continue

            paragraph = ''.join([
                rawAnswer[text['formFieldId']]
                for text in ratingText['texts']
                if text['formFieldId'] in rawAnswer
                and rawAnswer[text['formFieldId']]
            ]).strip(' \t\n\r')

            if len(paragraph) < 1:
                continue

            rating = rawAnswer[ratingText['rating']['formFieldId']]
            ratings.append(rating)
            paragraphs.append(paragraph)

        ratings = np.array(ratings, dtype=np.float)

        return (ratings, paragraphs)

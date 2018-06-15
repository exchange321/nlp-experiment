import json
import pprint

pp = pprint.PrettyPrinter(indent=4)


class Results:
    def __init__(self):
        self.results = []

    def add(self, result):
        self.results.append(result)

    def json(self):
        return json.dumps(
            [result.json() for result in self.results], indent=4
        )

    def __str__(self):
        output = ''
        for result in self.results:
            output += str(result)

        return output


class Result:
    def __init__(self, tenant, name, paragraphs, results={}):
        assert type(results) is dict
        self.results = {**results}

        self.tenant = tenant
        self.name = name
        self.paragraphs = paragraphs

    def add(self, name, result):
        self.results[name] = result

    def get(self, name):
        return self.results[name]

    def remove(self, name):
        del self.results[name]

    def json(self):
        return [
            {
                'name': self.tenant,
                'children': self.results[name].json(self.name)
            } for name in self.results
        ]
        return {
            'name': f'{self.tenant} - {self.name}',
            'children': self.results.json()
        }

    def __str__(self):
        output = ''
        output += '########################################'\
            '########################################\n'
        output += f'#####{self.tenant:^70}#####\n'
        output += f'#####{self.name:^70}#####\n'
        output += '########################################'\
            '########################################\n'
        output += '\n'

        paragraphTitle = 'Paragraphs'
        output += f'#####{paragraphTitle:^70}#####\n'
        output += '\n'
        for paragraph in self.paragraphs:
            for i in range(0, len(paragraph), 72):
                if i == 0:
                    output += f'  ==>\t{paragraph[i:i+72]}\n'
                else:
                    output += f'     \t{paragraph[i:i+72]}\n'
            output += '\n'

        output += f'#####{(f"Number of Comments: {len(self.paragraphs)}"):^70}#####\n'  # noqa: E501
        output += '\n'

        for name in self.results:
            result = self.results[name]
            output += f'#####{name:^70}#####\n'
            output += f'\n{result}\n'
        output += '########################################'\
            '########################################\n'
        output += '\n'
        return output

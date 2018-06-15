def output(results, filename=None, toJson=False):
    output = results.json() if toJson else str(results)

    if filename:
        f = open(filename, 'w+')
        f.write(output)
        f.close()

    print(output)

import csv

def begin_parse(input_file):
    with open(input_file) as f:
        reader = csv.reader(f, delimiter = '\t')
        rows = [row for row in reader]
        return rows

def pluck(objects, attribute):
    return map(lambda obj: getattr(obj, attribute), objects)



# coding: utf-8

# std
from datetime import datetime
import math
from export_events_to_csv_annotable import get_mabed

# mabed
import mabed.utils as utils

__author__ = "Corentin F."
__email__ = ""


def parse_date(d: str):
    return datetime.strptime(d, "%Y-%m-%d %H:%M:%S")


if __name__ == '__main__':
    mabed = get_mabed()
    mabed.run(k=100, p=10, theta=0.6, sigma=0.5)
    events = list(utils.iterate_events_as_dict(mabed))

    for e in [events[32]]:
        imp: list = list(e['impact'])
        while imp and imp[0]['value'] == 0:
            imp.pop(0)
        while imp and imp[-1]['value'] == 0:
            imp.pop(-1)

        first_date = parse_date(imp[0]['date'])
        last_date = parse_date(imp[-1]['date'])
        delta = last_date - first_date

        min_value = min(imp, key=lambda x: x['value'])['value']
        max_value = max(imp, key=lambda x: x['value'])['value']
        delta_v = max_value - min_value

        N = len(imp)

        def date_to_float01(d):
            x = (d - first_date).total_seconds() / delta.total_seconds()
            p = int(max(1, math.log10(N))) + 1
            x = round(x, p)
            return x

        def value_to_float01(v):
            return round((v - min_value) / delta_v, 2)

        for x in imp:
            d, v = x['date'], x['value']
            d = parse_date(d)
            d = date_to_float01(d)
            v = value_to_float01(v)
            # print(d, v, end=' ')
            print(x['date'], d, v)

        print('/')
        exit()

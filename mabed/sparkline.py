from datetime import datetime
import math


def parse_date(d: str):
    return datetime.strptime(d, "%Y-%m-%d %H:%M:%S")


def spark_event(event: dict):
    out = ''

    imp: list = list(event['impact'])
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
        out += f"{d} {v} "
        # out += " ".join(x['date'], d, v)

    out += "/"
    return out


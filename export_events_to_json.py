# coding: utf-8

# std
import time
import argparse
import json

# mabed
import mabed.utils as utils

__author__ = "Corentin F."
__email__ = ""

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Build event browser')
    p.add_argument('i', metavar='input', type=str, help='Input pickle file')
    p.add_argument('o', metavar='output', type=str,
                   help='Output json file', default=None)
    args = p.parse_args()

    print('Loading events from %s...' % args.i)
    mabed = utils.load_events(args.i)

    # format data
    print('Preparing data...')

    out_events = []
    formatted_dates = []

    for i in range(0, mabed.corpus.time_slice_count):
        formatted_dates.append(str(mabed.corpus.to_date(i)))

    for event in mabed.events:
        mag = event[0]
        main_term = event[2]
        raw_anomaly = event[4]
        anomalies = []
        time_interval = event[1]
        related_terms = []

        for related_term in event[3]:
            related_terms.append({
                'term': related_term[0],
                'mag': related_term[1],
            })

        for i in range(0, mabed.corpus.time_slice_count):
            value = 0

            if time_interval[0] <= i <= time_interval[1]:
                value = raw_anomaly[i]
                if value < 0:
                    value = 0

            anomalies.append({
                'date': str(formatted_dates[i]),
                'value': value,
            })
            # anomalies.append(value)

        out_events.append({
            'mag': mag,
            'start': str(mabed.corpus.to_date(time_interval[0])),
            'end': str(mabed.corpus.to_date(time_interval[1])),
            'term': main_term,
            'related': related_terms,
            'impact': anomalies,
        })

    if args.o is not None:
        data = {
            'events': out_events,
            # 'dates': formatted_dates,
        }
        json.dump(data, open(args.o, 'w'))

        print('Done.')

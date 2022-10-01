# coding: utf-8

# std
import sys
import argparse
import json
from export_events_to_csv_annotable import get_mabed

# mabed
import mabed.utils as utils

__author__ = "Corentin F."
__email__ = ""

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Build event browser')
    # p.add_argument('i', metavar='input', type=str, help='Input pickle file')
    p.add_argument('o', metavar='output', type=str,
                   help='Output json file', default=None)
    args = p.parse_args()

    # print('Loading events from %s...' % args.i, file=sys.stderr)
    # mabed = utils.load_events(args.i)
    mabed = get_mabed()
    mabed.run(k=100, p=10, theta=0.6, sigma=0.5)

    # format data
    print('Preparing data...', file=sys.stderr)

    events = list(utils.iterate_events_as_dict(mabed))

    if args.o is not None:
        with open(args.o, 'w') as f:
            json.dump(events, f)
        print('Done.', file=sys.stderr)
    else:
        print(json.dumps(events))

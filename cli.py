from datetime import datetime, timedelta
from math import floor

import mabed.utils as utils
from mabed.corpus import Corpus
from mabed.mabed import MABED


def timedelta_to_string_human(td: timedelta):
    if isinstance(td, (float, int)):
        td = timedelta(seconds=td)

    seconds = td.total_seconds()
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    millis = int(td.microseconds / 1000)

    if hours > 0:
        return '{:.0f}h {:.0f}m {:.0f}s'.format(hours, minutes, seconds)
    elif minutes > 0:
        return '{:.0f}m {:.0f}s'.format(minutes, seconds)
    elif floor(seconds) > 0:
        return '{:.0f}s'.format(seconds)
    else:
        return '{:.0f}ms'.format(millis)


def print_timing(timing):
    if 'total' in timing:
        print('Full request duration:', timedelta_to_string_human(timing['total']))
    print('  Raw MABED duration:', timedelta_to_string_human(timing['get_raw_mabed']))
    print('  Event detection duration:', timedelta_to_string_human(timing['event_detection']))
    print('    Discretize duration:', timedelta_to_string_human(timing['discretize']))
    print('    Run duration:', timedelta_to_string_human(timing['run']))
    print('  Iterate events duration:', timedelta_to_string_human(timing['iterate_events']))
    print('  Find articles duration:', timedelta_to_string_human(timing['find_articles']))


def get_raw_mabed(
    *,
    input_path,
    stopwords,
    min_absolute_frequency,
    max_relative_frequency,
    filter_date_after,
    **rest,
):
    my_corpus = Corpus(
        source_file_path=input_path,
        stopwords_file_path=stopwords,
        min_absolute_freq=min_absolute_frequency,
        max_relative_freq=max_relative_frequency,
        filter_date_after=filter_date_after,
    )
    mabed = MABED(my_corpus)
    return mabed


def compute_events(mabed: MABED, params):
    mabed.corpus.discretize(time_slice_length=params['time_slice_length'])
    return mabed.run(k=params['k'], p=params['p'], theta=params['theta'], sigma=params['sigma'])


def do_mabed(params):
    timing = {}

    with utils.timing('get_raw_mabed', timing):
        mabed = get_raw_mabed(**params)

    print('Running MABED...')
    with utils.timing('event_detection', timing):
        # compute_events(mabed, params)
        with utils.timing('discretize', timing):
            mabed.corpus.discretize(time_slice_length=params['time_slice_length'])

        with utils.timing('run', timing):
            mabed.run(k=params['k'], p=params['p'], theta=params['theta'], sigma=params['sigma'])

    with utils.timing('iterate_events', timing):
        events = list(utils.iterate_events_as_dict(mabed))

    with utils.timing('find_articles', timing):
        articles = mabed.find_articles_for_events(n_articles=params['n_articles'])
        for e, a in zip(events, articles):
            e['articles'] = a

    return events, timing, mabed


if __name__ == '__main__':
    import argparse
    import json
    import os
    import sys

    root = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Run MABED on a corpus.')
    parser.add_argument('input', help='Path to the corpus file.', default='../out2.csv')
    parser.add_argument('--output', help='Path to the output file.', default=sys.stdout)
    parser.add_argument('--stopwords', help='Path to the stopwords file.', default=f'{root}/stopwords/custom.txt')
    parser.add_argument('--maf', type=int, help='Minimum absolute frequency.', default=10)
    parser.add_argument('--mrf', type=float, help='Maximum relative frequency.', default=0.2)
    parser.add_argument('--filter_date_after', type=lambda s: datetime.strptime(s, '%Y-%m-%d'), help='Filter articles after this date.', default='2014-01-01')
    parser.add_argument('--tsl', type=int, help='Time slice length.', default=182 * 24 * 60)
    parser.add_argument('--k', type=int, help='Number of top events to detect (0 for auto).', default=0)
    parser.add_argument('--p', type=int, help='Number of words per event.', default=10)
    parser.add_argument('--theta', type=float, help='Theta parameter.', default=0.6)
    parser.add_argument('--sigma', type=float, help='Sigma parameter.', default=0.5)
    parser.add_argument('--n_articles', type=int, help='Number of articles to retrieve for each event.', default=1)
    parser.add_argument('--verbose', action='store_true', help='Verbose mode.')
    parser.add_argument('--as-graph', action='store_true', help='Export as graph for Cytoscape.')

    args = parser.parse_args()

    params = {
        'input_path': args.input,
        'output_path': args.output,
        'stopwords': args.stopwords,
        'min_absolute_frequency': args.maf,
        'max_relative_frequency': args.mrf,
        'filter_date_after': args.filter_date_after,
        'time_slice_length': args.tsl,
        'k': args.k,
        'p': args.p,
        'theta': args.theta,
        'sigma': args.sigma,
        'n_articles': args.n_articles,
        'verbose': args.verbose,
        'as_graph': args.as_graph,
    }

    if args.verbose:
        print('Running MABED with the following parameters:')
        for k, v in params.items():
            if v is not None:
                print('  {}: {}'.format(k, v))

    events, timing, mabed = do_mabed(params)

    if args.verbose:
        print_timing(timing)

    if args.as_graph:
        out = mabed.as_cytoscape_graph()
        if args.output == sys.stdout or args.output == '-':
            json.dump(out, sys.stdout, indent=2)
        else:
            with open(args.output, 'w') as f:
                json.dump(out, f, indent=2)

        sys.exit(0)

    if args.output == sys.stdout or args.output == '-':
        json.dump(events, sys.stdout, indent=2)
    else:
        with open(args.output, 'w') as f:
            json.dump(events, f, indent=2)

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
    mabed = MABED(my_corpus, extra=rest)
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
            mabed.run(k=params['k'], p=params['p'], theta=params['theta'], sigma=params['sigma'], event_filter=params.get('event_filter', None))

    with utils.timing('iterate_events', timing):
        events = list(utils.iterate_events_as_dict(mabed))

    if params.get('n_articles', 0) > 0:
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
    parser.add_argument('-o', '--output', help='Path to the output file.', default=sys.stdout)
    parser.add_argument('-w', '--stopwords', help='Path to the stopwords file.', default=f'{root}/stopwords/custom.txt')
    parser.add_argument('--maf', type=int, help='Minimum absolute frequency.', default=10)
    parser.add_argument('--mrf', type=float, help='Maximum relative frequency.', default=0.2)
    parser.add_argument('-D', '--filter_date_after', type=lambda s: datetime.strptime(s, '%Y-%m-%d'), help='Filter articles after this date.', default='2014-01-01')
    parser.add_argument('-L', '--tsl', type=int, help='Time slice length.', default=182 * 24 * 60)
    parser.add_argument('-k', '--k', type=int, help='Number of top events to detect (0 for auto).', default=0)
    parser.add_argument('-p', '--p', type=int, help='Number of words per event.', default=10)
    parser.add_argument('-t', '--theta', type=float, help='Theta parameter.', default=0.6)
    parser.add_argument('-s', '--sigma', type=float, help='Sigma parameter.', default=0.5)
    parser.add_argument('-N', '--n_articles', type=int, help='Number of articles to retrieve for each event.', default=1)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode.')
    parser.add_argument('-G', '--as-graph', action='store_true', help='Export as graph for Cytoscape.')
    parser.add_argument('-F', '--event-filter', type=str, help='Python code of the body of a function used to filter events. Its signature is (event) -> bool. Example: --event-filter=\'"coronavirus" in event.term.split(", ")\'')
    parser.add_argument('--json', action='store_true', help='Write output events as JSON')
    parser.add_argument('--cluster-func', type=str, help='Event distance function to use during clustering. It is a string composed of two parts: an aggregator (sum, prod, norm) and a set of properties (text, time, gap) separated by commas. Example: --cluster-func="norm:text,time,gap"')
    parser.add_argument('-x', '--option', type=str, action='append', help='Options to pass to the MABED "extra" parameter. It is a string composed of two parts: a key and a value separated by a colon. Example: --option="recursive_clustering:yes"')

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
        'cluster_func': args.cluster_func,
    }

    if args.event_filter:
        def event_filter__generated(events, mabed):
            def run_filter_one(event):
                event_id = event[utils.EVT.MAIN_TERM]
                time_interval = event[utils.EVT.TIME_INTERVAL]
                time_start = str(mabed.corpus.to_date(time_interval[0]))
                time_end = str(mabed.corpus.to_date(time_interval[1]))

                main_terms = utils.get_main_terms(event)
                related_terms = utils.get_related_terms(event)
                all_terms = main_terms | related_terms

                res = eval(args.event_filter, None, {
                    'event': event,
                    'id': event_id,
                    'mag': event[utils.EVT.MAG],
                    'main_terms': main_terms,
                    'related_terms': related_terms,
                    'all_terms': all_terms,
                    'time_start': time_start,
                    'time_end': time_end,
                })

                # if res:
                #     print('\x1b[32m', end='')
                #     print('    ☑︎', args.event_filter)
                # else:
                #     print('\x1b[2;31m', end='')
                #     print('    ☐', args.event_filter)
                # print('    │', 'main_terms:', main_terms)
                # print('    │', 'related_terms:', related_terms)
                # print('\x1b[0m', end='')

                return bool(res)

            return list(filter(run_filter_one, events))

        params['event_filter'] = event_filter__generated

    map_str_to_val = {
        'True': True,
        'False': False,
        }
    for option in args.option or []:
        if ':' in option:
            key, value = option.split(':')
            value = map_str_to_val.get(value, value)
            params[key] = value
        else:
            params[key] = True

    if args.verbose:
        print('Running MABED with the following parameters:')
        for k, v in params.items():
            if v is not None:
                print('  {}: {}'.format(k, v))

    events, timing, mabed = do_mabed(params)

    if args.verbose:
        print_timing(timing)

    if args.as_graph:
        g = mabed.as_cytoscape_graph()
        if args.output == sys.stdout or args.output == '-':
            json.dump(g, sys.stdout, indent=2)
        else:
            with open(args.output, 'w') as f:
                json.dump(g, f, indent=2)

        sys.exit(0)

    if args.json:
        if args.output == sys.stdout or args.output == '-':
            json.dump(events, sys.stdout, indent=2)
        else:
            with open(args.output, 'w') as f:
                json.dump(events, f)
    else:
        print(f'{len(events)} events:')
        for ev in events:
            print(utils.stringify_rich_event(ev, mabed, more_info=True))

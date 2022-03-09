"""
Simple Flask API server returning JSON data
"""

from datetime import datetime, timedelta
from math import floor
from operator import itemgetter
import os

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from export_events_to_csv_annotable import get_mabed
from mabed.mabed_cache import cached_getpath

import mabed.utils as utils
from mabed.mabed import MABED
from mabed.corpus import Corpus

app = Flask(__name__, static_folder='browser/static',
            template_folder='browser/templates')
CORS(app)


def timedelta_to_string_human(td: timedelta):
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


# def get_mabed(**kwargs) -> MABED:
#     # if os.path.exists(mabed_pickle_path):
#     #     mabed = utils.load_pickle(mabed_pickle_path)
#     #     for key, value in kwargs.items():
#     #         if (getattr(mabed, key, None) or getattr(mabed.corpus, key, None)) != value:
#     #             break
#     #     else:  # No break -> all attributes match
#     #         return mabed

#     # Else rebuild mabed object
#     mabed = init_mabed(**kwargs)
#     # utils.save_pickle(mabed, mabed_pickle_path)
#     return mabed


# @app.route('/api/events', methods=['GET'])
# def events():
#     if not request.json or any((
#         not 'k' in request.json,
#         not 'p' in request.json,
#         not 't' in request.json,
#         not 's' in request.json,
#     )):
#         return jsonify({'error': 'Invalid request'}), 400

#     # Load the model
#     params = get_default_params()
#     mabed = get_mabed(**params)

#     k = request.json['k']
#     p = request.json['p']
#     theta = request.json['t']
#     sigma = request.json['s']

#     mabed.run(k, p, theta, sigma)
#     events = list(utils.iterate_events_as_dict(mabed))
#     return jsonify(events)


def missing_param(param):
    return {'error': 'Missing parameter: ' + param}


@app.route('/api/events.json', methods=['GET'])
def events_GET():
    full_request_duration = datetime.now()

    # Retrieve GET parameters
    path = request.args.get('path', default='stock_article.csv', type=str)
    stopwords = request.args.get(
        'stopwords', default='stopwords/twitter_en.txt', type=str)

    maf = request.args.get('maf', default=10, type=int)
    mrf = request.args.get('mrf', default=0.4, type=float)

    tsl = request.args.get('tsl', default=24*60, type=int)

    k = request.args.get('k', type=int)
    p = request.args.get('p', default=10, type=int)
    theta = request.args.get('t', default=0.6, type=float)
    sigma = request.args.get('s', default=0.6, type=float)
    filter_date_after = request.args.get(
        'from_date', default="2019-01-01", type=str)
    filter_date_after = datetime.strptime(filter_date_after, '%Y-%m-%d')

    n_articles = request.args.get('n_articles', default=3, type=int)

    if k is None:
        return jsonify(missing_param('k')), 400

    params = {}

    params['label'] = request.args.get('label', default='no label', type=str)

    params['min_absolute_frequency'] = maf
    params['max_relative_frequency'] = mrf
    params['input_path'] = path
    params['stopwords'] = stopwords
    params['filter_date_after'] = filter_date_after

    params['time_slice_length'] = tsl
    params['k'] = k
    params['p'] = p
    params['theta'] = theta
    params['sigma'] = sigma

    raw_mabed_duration = datetime.now()
    mabed = get_raw_mabed(**params)
    raw_mabed_duration = datetime.now() - raw_mabed_duration

    print('Running MABED...')
    with utils.timer('Event detection performed'):
        compute_events_duration = datetime.now()
        # compute_events(mabed, params)
        discretize_events_duration = datetime.now()
        mabed.corpus.discretize(time_slice_length=params['time_slice_length'])
        discretize_events_duration = datetime.now() - discretize_events_duration

        mabed_run_duration = datetime.now()
        mabed.run(k=params['k'], p=params['p'],
                  theta=params['theta'], sigma=params['sigma'])
        mabed_run_duration = datetime.now() - mabed_run_duration

        compute_events_duration = datetime.now() - compute_events_duration

    iterate_events_duration = datetime.now()
    events = list(utils.iterate_events_as_dict(mabed))
    iterate_events_duration = datetime.now() - iterate_events_duration

    find_articles_duration = datetime.now()
    articles = mabed.find_articles_for_events(n_articles=n_articles)
    for e, a in zip(events, articles):
        e['articles'] = a
    find_articles_duration = datetime.now() - find_articles_duration

    res = jsonify(events)

    full_request_duration = datetime.now() - full_request_duration

    return res


# @app.route('/')
# def index():
#     return send_from_directory('html', 'index.html')


@app.route('/')
def index():
    return render_template('empty.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)

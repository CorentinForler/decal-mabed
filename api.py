"""
Simple Flask API server returning JSON data
"""

import os

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import mabed.utils as utils
from mabed.mabed import MABED
from mabed.corpus import Corpus

mabed_pickle_path = 'out.pickle'

app = Flask(__name__, static_folder='browser/static',
            template_folder='browser/templates')
CORS(app)


def get_default_params():
    return {
        # BE CAREFUL, those variables
        # can be used to retrieve files
        # like the /etc/shadow file ...

        'input_path': 'stock_article.csv',
        # 'input_path': 'stock_article_20000.csv',
        'stopwords': 'customStopWords.txt',
        'csv_separator': '\t',

        'min_absolute_frequency': 10,
        'max_relative_frequency': 0.4,
        'time_slice_length': 24*60,
        'keep_corpus': False,
    }


def init_mabed(
    input_path='stock_article_20000.csv',
    stopwords='customStopWords.txt',
    csv_separator='\t',
    min_absolute_frequency=10,
    max_relative_frequency=0.4,
    time_slice_length=24*60,
    keep_corpus=False,
):
    my_corpus = Corpus(
        input_path,
        stopwords,
        min_absolute_frequency,
        max_relative_frequency,
        csv_separator)

    if keep_corpus:
        # my_corpus.import_discretized('discretized.pickle', time_slice_length)
        pass
    else:
        my_corpus.discretize(time_slice_length)

    mabed = MABED(my_corpus)
    return mabed


def get_mabed(**kwargs) -> MABED:
    if os.path.exists(mabed_pickle_path):
        mabed = utils.load_pickle(mabed_pickle_path)
        for key, value in kwargs.items():
            if (getattr(mabed, key, None) or getattr(mabed.corpus, key, None)) != value:
                break
        else:  # No break -> all attributes match
            return mabed

    # Else rebuild mabed object
    mabed = init_mabed(**kwargs)
    utils.save_pickle(mabed, mabed_pickle_path)
    return mabed


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
    # Retrieve GET parameters
    maf = request.args.get('maf', default=10, type=int)
    mrf = request.args.get('mrf', default=0.4, type=float)

    tsl = request.args.get('tsl', default=24*60, type=int)

    k = request.args.get('k', type=int)
    p = request.args.get('p', default=10, type=float)
    theta = request.args.get('t', default=0.6, type=float)
    sigma = request.args.get('s', default=0.6, type=float)

    if k is None:
        return jsonify(missing_param('k')), 400

    # Load the model
    print('Loading MABED...')
    with utils.timer('MABED loaded'):
        params = get_default_params()

        params['min_absolute_frequency'] = maf
        params['max_relative_frequency'] = mrf
        params['time_slice_length'] = tsl

        mabed = get_mabed(**params)

    print('Running MABED...')
    with utils.timer('Event detection performed'):
        mabed.run(k, p, theta, sigma)

    utils.save_pickle(mabed, mabed_pickle_path)

    events = list(utils.iterate_events_as_dict(mabed))
    return jsonify(events)


@app.route('/')
def index():
    return send_from_directory('html', 'index.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)

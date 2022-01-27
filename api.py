"""
Simple Flask API server returning JSON data
"""

from email.policy import default
import os

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS

import mabed.utils as utils
from mabed.mabed import MABED
from mabed.corpus import Corpus

mabed_pickle_path = 'out.pickle'

app = Flask(__name__, static_folder='browser/static',
            template_folder='browser/templates')
CORS(app)


def init_mabed(
    *,
    input_path,
    stopwords,
    min_absolute_frequency,
    max_relative_frequency,
    time_slice_length,
):
    my_corpus = Corpus(
        input_path,
        stopwords,
        min_absolute_frequency,
        max_relative_frequency)

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
    path = request.args.get('path', default='stock_article.csv', type=str)
    stopwords = request.args.get(
        'stopwords', default='stopwords/twitter_en.txt', type=str)

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
        params = {}

        params['min_absolute_frequency'] = maf
        params['max_relative_frequency'] = mrf
        params['time_slice_length'] = tsl
        params['input_path'] = path
        params['stopwords'] = stopwords

        mabed = get_mabed(**params)

    print('Running MABED...')
    with utils.timer('Event detection performed'):
        mabed.run(k, p, theta, sigma)

    utils.save_pickle(mabed, mabed_pickle_path)

    events = list(utils.iterate_events_as_dict(mabed))
    return jsonify(events)


# @app.route('/')
# def index():
#     return send_from_directory('html', 'index.html')


@app.route('/')
def index():
    return render_template('empty.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)

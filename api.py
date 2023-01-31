"""
Simple Flask API server returning JSON data
"""

from datetime import datetime
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import json

from cli import do_mabed, print_timing

app = Flask(__name__, static_folder='browser/static',
            template_folder='browser/templates')
CORS(app)


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
    mrf = request.args.get('mrf', default=0.2, type=float)

    tsl = request.args.get('tsl', default=24 * 60, type=int)

    k = request.args.get('k', type=int)
    p = request.args.get('p', default=10, type=int)
    theta = request.args.get('t', default=0.6, type=float)
    sigma = request.args.get('s', default=0.5, type=float)
    filter_date_after = request.args.get(
        'from_date', default="2019-01-01", type=str)
    filter_date_after = datetime.strptime(filter_date_after, '%Y-%m-%d')

    n_articles = request.args.get('n_articles', default=1, type=int)

    extra = request.args.get('extra', default="{}", type=str)
    try:
        extra = json.loads(extra)
    except:
        return jsonify({ "error": "invalid param: extra" }), 400

    if k is None:
        return jsonify(missing_param('k')), 400

    params = {}
    params.update(extra)

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

    params['n_articles'] = n_articles

    events, timing, mabed = do_mabed(params)

    res = jsonify(events)

    full_request_duration = datetime.now() - full_request_duration
    timing['total'] = full_request_duration.total_seconds()

    print_timing(timing)

    print('\x1b[6;30;42m' + 'SUCCESS' + '\x1b[0m')

    return res


# @app.route('/')
# def index():
#     return send_from_directory('html', 'index.html')


@app.route('/')
def index():
    return render_template('empty.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)

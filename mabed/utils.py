# coding: utf-8
import csv
from datetime import datetime
import timeit
import contextlib
import os
import pickle
from typing import Counter

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def guess_datetime_format(s: str):
    formats = [
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%Y.%m.%d',

        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',

        '%Y-%m-%d %H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S%Z',

        '%Y-%m-%d %H:%M:%S.%fZ',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%a, %d %b %Y %H:%M:%S %Z',
        '%a %d %b %Y %H:%M:%S %Z',
        '%a, %d %b %Y %H:%M:%S',
        '%a %d %b %Y %H:%M:%S',
    ]
    for fmt in formats:
        try:
            datetime.strptime(s, fmt)
            return fmt
        except ValueError:
            continue
    return None

# def guess_datetime_format(s: str) -> str:
#     fmt = ''
#     if len(s) <= 10:
#         fmt = '%Y-%m-%d'
#     else:
#         fmt = '%Y-%m-%d %H:%M:%S'
#     if '/' in s:
#         fmt = fmt.replace('-', '/', 2)
#     elif '.' in s:
#         fmt = fmt.replace('-', '.', 2)
#     try:
#         datetime.strptime(s, fmt)
#         return fmt
#     except ValueError:
#         return None


def save_events(mabed_object, file_path):
    with open(file_path, 'wb') as output_file:
        pickle.dump(mabed_object, output_file)


def load_events(file_path):
    with open(file_path, 'rb') as input_file:
        return pickle.load(input_file)


def write_vocabulary(vocabulary):
    with open('vocabulary.pickle', 'wb') as output_file:
        pickle.dump(vocabulary, output_file)


def load_stopwords(file_path):
    stopwords = set()
    with open(file_path, 'r') as input_file:
        for line in input_file.readlines():
            stopwords.add(line.strip('\n'))
    return stopwords


def timeslice_path(time_slice: int):
    return 'corpus/' + str(time_slice)


def append_to_time_slice(time_slice: int, tweet_text: str):
    with open(timeslice_path(time_slice), 'a', encoding='utf8') as time_slice_file:
        time_slice_file.write(tweet_text + '\n')


def init_time_slice(time_slice: int):
    """
    Creates or empties a time slice file.
    """
    with open(timeslice_path(time_slice), 'w', encoding='utf8') as dummy_file:
        dummy_file.write('')


def read_time_slice(time_slice: int):
    """
    Iterates over the tweets in a time slice.
    """
    with open(timeslice_path(time_slice), 'r', encoding='utf8') as time_slice_file:
        for tweet_text in time_slice_file:
            yield tweet_text.strip('\n')


def time_slice_exists(time_slice: int):
    """
    Returns True if the time slice exists and it is not empty (at least two bytes, arbitrarily).
    """
    return os.path.exists(timeslice_path(time_slice)) and os.stat(timeslice_path(time_slice)).st_size > 2


def load_pickle(file_path):
    with open(file_path, 'rb') as input_file:
        return pickle.load(input_file)


def save_pickle(data, file_path):
    with open(file_path, 'wb') as output_file:
        pickle.dump(data, output_file)


def pick_fields(obj, fields):
    """
    Returns a new object with only the fields in fields.
    """
    return {field: getattr(obj, field) for field in fields}


def iterate_events_as_dict(mabed):
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

        yield {
            'mag': mag,
            'start': str(mabed.corpus.to_date(time_interval[0])),
            'end': str(mabed.corpus.to_date(time_interval[1])),
            'term': main_term,
            'related': related_terms,
            'impact': anomalies,
        }


@contextlib.contextmanager
def timer(step_name: str):
    start_time = timeit.default_timer()
    yield start_time
    elapsed = timeit.default_timer() - start_time
    print(step_name, f'in {elapsed:.3f} seconds.')


def auto_detect_csv_settings(path: str):
    candidate_col_names_datetime = [
        'date', 'time', 'datetime', 'timestamp', 'date_time']
    candidate_col_names_text = [
        'text', 'tweet', 'tweet_text', 'message', 'content']

    separator = '\t'
    date_col_name = 'date'
    text_col_name = 'text'
    datetime_format = '%Y-%m-%d %H:%M:%S'

    with open(path, 'r', encoding='utf8') as f:
        header_line = f.readline()

        if '\t' in header_line:
            separator = '\t'
        elif ',' in header_line:
            separator = ','
        else:
            raise Exception('Failed to detect CSV SEPARATOR')

        f.seek(0)
        reader = csv.reader(f, delimiter=separator)
        header_line = next(reader)

        for c in header_line:
            if c.lower() in candidate_col_names_datetime:
                date_col_name = c
                break
        else:
            raise Exception('Failed to detect CSV DATE column name')

        for c in header_line:
            if c.lower() in candidate_col_names_text:
                text_col_name = c
                break
        else:
            raise Exception('Failed to detect CSV TEXT column name')

        # Grab ten lines or less
        date_index = header_line.index(date_col_name)
        dt_formats = Counter()
        i = 0
        for line in reader:
            dt = line[date_index]
            dt_fmt = guess_datetime_format(dt) or 'FAIL'
            dt_formats[dt_fmt] += 1
            i += 1
            if i > 20:
                break

        datetime_format = dt_formats.most_common(1)[0][0]
        if datetime_format == 'FAIL':
            raise Exception('Failed to detect CSV DATETIME FORMAT')

    print('   CSV guessed separator:', separator.replace('\t', '\\t'))
    print('   CSV guessed date column name:', date_col_name)
    print('   CSV guessed text column name:', text_col_name)
    print('   CSV guessed datetime format:', datetime_format)
    return (separator, date_col_name, text_col_name, datetime_format)

# coding: utf-8
import timeit
import contextlib
import os
import pickle

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


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

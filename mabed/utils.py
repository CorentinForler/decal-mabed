# coding: utf-8
import csv
import os
import sys
from datetime import datetime
import timeit
import contextlib
import pickle
import math
from typing import Counter
from operator import itemgetter

from mabed.sparkline import spark_event

__author__ = "Adrien Guille, Corentin F."
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


# def timeslice_path(time_slice: int):
#     return 'corpus/' + str(time_slice)
# def append_to_time_slice(time_slice: int, tweet_text: str):
#     with open(timeslice_path(time_slice), 'a', encoding='utf8') as time_slice_file:
#         time_slice_file.write(tweet_text + '\n')
# def init_time_slice(time_slice: int):
#     """Creates or empties a time slice file."""
#     with open(timeslice_path(time_slice), 'w', encoding='utf8') as dummy_file:
#         dummy_file.write('')
# def read_time_slice(time_slice: int):
#     """Iterates over the tweets in a time slice."""
#     with open(timeslice_path(time_slice), 'r', encoding='utf8') as time_slice_file:
#         for tweet_text in time_slice_file:
#             yield tweet_text.strip('\n')
# def time_slice_exists(time_slice: int):
#     """Returns True if the time slice exists and it is not empty (at least two bytes, arbitrarily)."""
#     return os.path.exists(timeslice_path(time_slice)) and os.stat(timeslice_path(time_slice)).st_size > 2


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


class EVT:
    MAG = 0
    TIME_INTERVAL = 1
    MAIN_TERM = 2
    RELATED_TERMS = 3
    ANOMALY = 4
    EXTRA = 5


class REL_TERM:
    TXT = 0
    MAG = 1


def event_tuple_as_dict(mabed, event):
    mag = event[EVT.MAG]
    main_term = event[EVT.MAIN_TERM]
    main_terms = set(main_term.split(', '))
    raw_anomaly = event[EVT.ANOMALY]
    anomalies = []
    time_interval = event[EVT.TIME_INTERVAL]
    related_terms = []

    for related_term in event[EVT.RELATED_TERMS]:
        if related_term[REL_TERM.TXT] in main_terms:
            continue
        related_terms.append({
            'term': related_term[REL_TERM.TXT],
            'mag': related_term[REL_TERM.MAG],
            # 'isMain': related_term[0] in main_term.split(', '),
        })

    for i in range(0, mabed.corpus.time_slice_count):
        value = 0

        if time_interval[0] <= i <= time_interval[1]:
            value = raw_anomaly[i]
            if value < 0:
                value = 0

        anomalies.append({
            'date': str(mabed.corpus.to_date(i)),
            'value': value,
        })

    return {
        'mag': mag,
        'start': str(mabed.corpus.to_date(time_interval[0])),
        'end': str(mabed.corpus.to_date(time_interval[1])),
        'term': main_term,
        'related': related_terms,
        'impact': anomalies,
        'extra': event[EVT.EXTRA] if len(event) > EVT.EXTRA else None,
    }


def iterate_events_as_dict(mabed):
    for event in mabed.events:
        yield event_tuple_as_dict(mabed, event)


@contextlib.contextmanager
def timer(step_name: str):
    start_time = timeit.default_timer()
    yield start_time
    elapsed = timeit.default_timer() - start_time
    print(step_name, f'in {elapsed:.3f} seconds.')


@contextlib.contextmanager
def timing(key: str, out: dict):
    start_time = timeit.default_timer()
    yield
    elapsed = timeit.default_timer() - start_time
    out[key] = elapsed


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


def parallel_iterator_unordered(it):
    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool()
    for i in pool.imap_unordered(lambda x: x, it):
        yield i


def getAllTermsForEvent(ev):
    terms = ev['term'].split(', ')
    for a in ev['related']:
        terms.extend(a['term'].split(', '))
    return list(set(terms))


def get_main_terms(event):
    return set(event[EVT.MAIN_TERM].split(', '))


def get_related_terms(event):
    return set(map(itemgetter(REL_TERM.TXT), event[EVT.RELATED_TERMS]))


def get_main_and_related_terms(event):
    return get_main_terms(event) | get_related_terms(event)


def millify(n):
    millnames = ['','k','·10⁶','·10⁹','·10¹²']
    # https://stackoverflow.com/a/3155023
    n = float(n)
    millidx = max(0, min(len(millnames)-1, int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    return '{:.2f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


def stringify_rich_event(event, mabed, more_info=False, is_terminal=sys.stdout.isatty(), use_iterm_annotations=('iTerm' in os.getenv('TERM_PROGRAM'))):
    out = []

    def pr(*args, end='\n', **kwargs):
        out.append(''.join(map(str, args)) + end)

    related_words = []
    for r in sorted(event["related"], key=itemgetter("mag"), reverse=True):
        related_word, weight = r["term"], r["mag"]
        related_words.append(f'{related_word} ({100*weight:.0f}%)')

    comma = is_terminal and "\x1b[2m,\x1b[0m " or ", "
    pr(f'┌╴ {comma.join(sorted(event["term"].split(", ")))}')
    pr(f'│ • Mag:   {millify(event["mag"])}')
    pr(f'│ • Start: {event["start"]}')
    pr(f'│ • End:   {event["end"]}')

    # Impact
    impact_min_dt = event["impact"][0]["date"]
    impact_min_val = event["impact"][0]["value"]
    impact_max_dt = event["impact"][0]["date"]
    impact_max_val = event["impact"][0]["value"]
    for x in event["impact"]:
        if x["value"] > impact_max_val:
            impact_max_dt = x["date"]
            impact_max_val = x["value"]
        if x["value"] < impact_min_val:
            impact_min_dt = x["date"]
            impact_min_val = x["value"]
    pr(f'│ • Impact:')
    pr(f'│   ├╴ min: {millify(impact_min_val):3} @ {impact_min_dt}')
    pr(f'│   └╴ max: {millify(impact_max_val):3} @ {impact_max_dt}')

    pr(f'│ • Related: {comma.join(related_words)}')
    # for i, r in enumerate(related_words):
    #     c = '└' if (i == len(related_words) - 1) else '├'
    #     pr(f'│   {c}╴ {r}')

    if more_info:
        spark = spark_event(event)
        if use_iterm_annotations:
            pr(f'│ • Sparkline: \x1b]1337;AddHiddenAnnotation=60 | {spark}\x07{spark[:60]}…')
        else:
            pr(f'│ • Sparkline: {spark}')

        if articles := event.get("articles", None):
            pr('│ • Articles:')
            for i, article in enumerate(sorted(articles, key=itemgetter(0), reverse=True)):
                art_mag, art_text = article
                c = '└' if (i == len(articles) - 1) else '├'
                if use_iterm_annotations:
                    pr(f'│   {c}╴ \x1b]1337;AddHiddenAnnotation=60 | {art_text}\x07{art_text[:60]}…')
                else:
                    pr(f'│   {c}╴ {art_text}')

        if extra := event.get("extra", None):
            pr('│ • Extra:')
            if cluster := extra.get("cluster", None):
                pr(f'│   ├╴ Source cluster size: {len(cluster)}')
                for i, subev in enumerate(cluster):
                    # c = '└╴' if (i == len(cluster) - 1) else '├╴'
                    # pr(f'│   {c}╴ {subev[EVT.MAIN_TERM]}')
                    c = '• '
                    rich_subev = event_tuple_as_dict(mabed, subev)
                    formatted_subev = stringify_rich_event(rich_subev, mabed, more_info=more_info, is_terminal=is_terminal, use_iterm_annotations=use_iterm_annotations)
                    for line in formatted_subev.splitlines():
                        pr(f'│   {c} {line}')
                        c = '│ '

            # pr(f'│   └╴', '[JSON data...]')
            # pr(f'│   └╴', json.dumps(event["extra"]))
            # pr(f'│   └╴')

    pr('└╴')

    return ''.join(out)

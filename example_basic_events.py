from datetime import datetime

# mabed
from mabed.corpus import Corpus
from mabed.mabed import MABED
from mabed.mabed_cache import CacheLevel, cached_getpath


def get_corpus():
    input_path = "../stock_article.csv"
    stopwords = "./stopwords/custom.txt"
    min_absolute_frequency = 10
    max_relative_frequency = 0.2
    time_slice_length = 1440
    filter_date_after = datetime.strptime("2020-01-01", '%Y-%m-%d')

    corpus = Corpus(
        source_file_path=input_path,
        stopwords_file_path=stopwords,
        min_absolute_freq=min_absolute_frequency,
        max_relative_freq=max_relative_frequency,
        filter_date_after=filter_date_after,
    )
    corpus.discretize(time_slice_length)
    return corpus


def get_mabed(corpus: Corpus = None):
    if corpus is None:
        corpus = get_corpus()
    mabed = MABED(corpus)
    return mabed


def detect_events(mabed: MABED):
    # Perform full event detection
    n_events_to_detect = 10
    max_keywords_per_event = 10
    mabed.run(
        k=n_events_to_detect,
        p=max_keywords_per_event,
        theta=0.6, sigma=0.5)
    mabed.print_events()

    # cache_key = cached_getpath(
    #     mabed.corpus, CacheLevel.L1_DATASET, filename='', ext='', mabed=mabed)
    # print('Cache Key:')
    # print(cache_key)
    # print()


def find_articles(mabed: MABED):
    if not mabed.events:
        detect_events(mabed)

    n_articles = 1
    articles = mabed.find_articles_for_events(n_articles=n_articles)
    for i, event in enumerate(mabed.events):
        mabed.print_event(event)
        for score, text in articles[i]:
            print("|->", text)
        print()
        print()


def get_basic_events(mabed: MABED):
    basic_events_path = cached_getpath(
        mabed.corpus, CacheLevel.L3_DISCRETE, filename='basic_events', ext='.json', mabed=mabed)
    print('Basic Events JSON File Path:')
    print(basic_events_path)

    print()
    print("- " * 6)
    print()

    basic_events = mabed.phase1()
    print(len(basic_events), 'basic events detected')
    return basic_events


if __name__ == '__main__':
    mabed = get_mabed()
    basic_events = get_basic_events(mabed)
    magnitude, max_interval, vocab_entry0, anomaly = basic_events[0]

    print('Basic Event #1')
    print('* Word:', vocab_entry0)
    print('* Magnitude:', magnitude)
    print('* Max time interval:', max_interval)
    # print('* Anomaly:', anomaly)

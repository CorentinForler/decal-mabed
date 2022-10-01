from datetime import datetime

# mabed
from mabed.corpus import Corpus
from mabed.mabed import MABED
from mabed.mabed_cache import CacheLevel, cached_getpath
import mabed.stats as st
import numpy as np
import functools as f
import sklearn.cluster as cl
import json
import time
from tqdm import tqdm
import os

getRelatedWordsMemoize = dict()
my_inf = 123456

def get_corpus():
    input_path = "../treated_modified_start_from_03_2019.csv"
    stopwords = "./stopwords/custom.txt"
    min_absolute_frequency = 10
    max_relative_frequency = 0.2
    time_slice_length = 1440
    filter_date_after = datetime.strptime("2019-01-01", '%Y-%m-%d')

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


def getHashEvent(ev1):
    return hash(ev1[0]) + hash(ev1[2])

def get_related_words(ev1, mabed: MABED):
    hash = getHashEvent(ev1)
    if hash in getRelatedWordsMemoize:
        return getRelatedWordsMemoize[hash]

    main_word = ev1[2]
    candidate_words = mabed.corpus.cooccurring_words(
        ev1, mabed.p)
    main_word_freq = mabed.corpus.global_freq[mabed.corpus.vocabulary[main_word], :].toarray(
        )
    main_word_freq = main_word_freq[0, :]
    related_words = []
    if candidate_words is not None:
        for candidate_word in candidate_words:
            candidate_word_freq = mabed.corpus.global_freq[mabed.corpus.vocabulary[candidate_word], :].toarray(
            )
            candidate_word_freq = candidate_word_freq[0, :]

            # compute correlation and filter according to theta
            weight = st.erdem_correlation(
                main_word_freq, candidate_word_freq, ev1[1][0], ev1[1][1])
            if weight >= mabed.theta:
                related_words.append((candidate_word, weight))

    getRelatedWordsMemoize[hash] = related_words

    return related_words

def compute_distance_matrix(basic_events, mabed: MABED):
    distance_matrix = np.ones((len(basic_events), len(basic_events)), dtype=np.double) * my_inf

    for i in tqdm(range(len(basic_events))):
        for j in range(len(basic_events)):
            if i != j and distance_matrix[i][j] == my_inf:
                dist = dist_event(basic_events[i], basic_events[j], mabed)
                np.put(distance_matrix[i], j, dist)
                np.put(distance_matrix[j], i, dist)
            elif i == j:
                np.put(distance_matrix[i], j, 0)

    return distance_matrix

def dist_event(ev1, ev2, mabed : MABED):
    related_words_ev1 = get_related_words(ev1, mabed)
    related_words_ev2 = get_related_words(ev2, mabed)

    min_nb_word = min(len(related_words_ev1), len(related_words_ev2))

    main_word_1 = ev1[2]
    main_word_2 = ev2[2]

    main_word = (main_word_1 == main_word_2)

    if min_nb_word == 0:
        return my_inf

    count_word = 0
    for word, weight in related_words_ev1:
        for word2, weight2 in related_words_ev2:
            if word == word2:
                count_word += 1
                break

    return 1 - ((main_word/3) +  2*((count_word / min_nb_word)/3)) * st.overlap_coefficient(ev1[1], ev2[1])



if __name__ == '__main__':
    start_time = time.time()

    mabed = get_mabed()
    basic_events = get_basic_events(mabed)
    magnitude, max_interval, vocab_entry0, anomaly = basic_events[0]
    # anomaly is a time series

    print('Basic Event #1')
    print('* Word:', vocab_entry0)
    print('* Magnitude:', magnitude)
    print('* Max time interval:', max_interval)
    # print('* Anomaly:', anomaly)

    # basic_event = basic_events[0]
    # main_word = basic_event[2]

    mabed.p = 10
    mabed.theta = 0.6

    # candidate_words = mabed.corpus.cooccurring_words(
    #     basic_event, mabed.p)
    # main_word_freq = mabed.corpus.global_freq[mabed.corpus.vocabulary[main_word], :].toarray(
    # )

    # print('p:', mabed.p)
    # print('Candidate words:', candidate_words)
    # print('main word frequency', main_word_freq)

    if os.path.exists("distance_matrix.json"):
        with open("distance_matrix.json", "r") as f:
            distance_matrix = json.load(f)
    else:
        distance_matrix = None

    # dist = dist_event(basic_events[0], basic_events[1], mabed)

    if distance_matrix is None:
        distance_matrix = compute_distance_matrix(basic_events, mabed)
        print(distance_matrix)

        with open("distance_matrix.json", "w") as file_dm:
            json.dump(distance_matrix.tolist(), file_dm)

    clustering = cl.OPTICS(min_samples=2, max_eps=0.4, metric="precomputed", n_jobs=-1).fit(distance_matrix)
    print(clustering.labels_)
    print("number of clusters : ", set(clustering.labels_))
    for i in range(len(list(set(clustering.labels_)))):
        print("number in clusters : ", list(set(clustering.labels_))[i], " : ", clustering.labels_.tolist().count(list(set(clustering.labels_))[i]))

    end_time = time.time()
    print("duration:", end_time - start_time)

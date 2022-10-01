from datetime import datetime
from operator import itemgetter
import itertools

import mabed.stats as st
from mabed.corpus import Corpus
from mabed.mabed import MABED
import mabed.utils as utils


my_inf = 1e99


def get_corpus():
    input_path = "../out2.csv"
    stopwords = "./stopwords/custom.txt"
    min_absolute_frequency = 10
    max_relative_frequency = 0.2
    time_slice_length = 262080
    filter_date_after = datetime.strptime("2014-01-01", '%Y-%m-%d')

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

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def dist_event(ev1, ev2, mabed : MABED):
    related_words_ev1 = set(map(itemgetter(0), ev1[3]))
    related_words_ev2 = set(map(itemgetter(0), ev2[3]))

    min_nb_word = min(len(related_words_ev1), len(related_words_ev2))

    if min_nb_word == 0:
        return my_inf

    main_words_1 = ev1[2].split(', ')
    main_words_2 = ev2[2].split(', ')

    share_a_main_word = len(intersection(main_words_1, main_words_2)) > 0

    count_word = len(intersection(related_words_ev1, related_words_ev2))

    return 1 - ((1 / 3) * share_a_main_word + (2 / 3) * (count_word / min_nb_word)) * st.overlap_coefficient(ev1[1], ev2[1])

def dist_event3(ev1, ev2, mabed : MABED):
    words_ev1 = set(map(itemgetter(0), ev1[3]))
    words_ev2 = set(map(itemgetter(0), ev2[3]))

    main_words_1 = ev1[2].split(', ')
    main_words_2 = ev2[2].split(', ')

    words_ev1.update(main_words_1)
    words_ev2.update(main_words_2)

    all_words = words_ev1.union(words_ev2)

    prop_shared_words = len(intersection(words_ev1, words_ev2)) / len(all_words)
    # assert 0 <= prop_shared_words <= 1

    return 1 - prop_shared_words * st.overlap_coefficient(ev1[1], ev2[1])

def dist_event2(ev1, ev2, mabed : MABED):
    # related_words_ev1 = get_related_words(ev1, mabed)
    # related_words_ev2 = get_related_words(ev2, mabed)

    related_words_ev1 = []
    related_words_ev2 = []

    for word, _ in ev1[3]:
        related_words_ev1.append(word)

    for word, _ in ev2[3]:
        related_words_ev2.append(word)

    min_nb_word = min(len(related_words_ev1), len(related_words_ev2))

    if min_nb_word == 0:
        return my_inf

    main_words_1 = ev1[2].split(', ')
    main_words_2 = ev2[2].split(', ')

    main_word = (len(intersection(main_words_1, main_words_2)) > 0)

    count_word = 0
    for word in related_words_ev1:
        for word2 in related_words_ev2:
            if word == word2:
                count_word += 1
                break

    return 1 - ((main_word/3) +  2*((count_word / min_nb_word)/3)) * st.overlap_coefficient(ev1[1], ev2[1])

def get_events() -> list:
    from mabed.mabed_cache import _cached_load
    file_path = "/Users/cogk/Desktop/phd-track-event-vis-with-optics/phd-track-event-vis/cache/b05839acef0da2db09ec705b0a02fa8e_2014-01-01/maf=10,mrf=0.2/tsl=262080/basic_events.json"
    with open(file_path, "rb") as file:
        return _cached_load(file, file_path)

def main():
    mabed = get_mabed()

    # Paramètres
    nb_ev_max = 100 # Nombre d'events que MABED doit trouver
    nb_ev_tgt = 25 # Nombre d'events que l'utilisateur souhaite récupérer

    p = 10
    theta = 0.6
    sigma = 0.5

    mabed.run(nb_ev_max, p, theta, sigma)

    for i, j in itertools.combinations(range(len(mabed.events)), 2):
        ev1 = mabed.events[i]
        ev2 = mabed.events[j]
        assert dist_event(ev1, ev2, mabed) == dist_event2(ev1, ev2, mabed)

    with utils.timer("Method 1"):
        for i, j in itertools.combinations(range(len(mabed.events)), 2):
            ev1 = mabed.events[i]
            ev2 = mabed.events[j]
            dist_event(ev1, ev2, mabed)

    with utils.timer("Method 2"):
        for i, j in itertools.combinations(range(len(mabed.events)), 2):
            ev1 = mabed.events[i]
            ev2 = mabed.events[j]
            dist_event2(ev1, ev2, mabed)

    with utils.timer("Method 3"):
        for i, j in itertools.combinations(range(len(mabed.events)), 2):
            ev1 = mabed.events[i]
            ev2 = mabed.events[j]
            dist_event3(ev1, ev2, mabed)

    print(mabed.events[0])
    print(next(utils.iterate_events_as_dict(mabed)))


main()


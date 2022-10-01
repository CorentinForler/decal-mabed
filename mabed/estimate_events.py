import itertools
from typing import TYPE_CHECKING
import numpy as np
from tqdm import tqdm
import sklearn.cluster as cl
from collections import Counter
from operator import itemgetter
import math

if TYPE_CHECKING:
    from mabed.mabed import MABED

from mabed.mabed_cache import CacheLevel, mabed_cached
import mabed.stats as st
from mabed.utils import EVT


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def flatmap(func, *iterable):
    return itertools.chain.from_iterable(map(func, *iterable))


my_infinity = 1e99

def dist_event(ev1, ev2):
    # Collect the main terms of the two events
    main_terms_1 = set(ev1[EVT.MAIN_TERM].split(', '))
    main_terms_2 = set(ev2[EVT.MAIN_TERM].split(', '))

    # Collect the related terms of the two events
    all_terms_1 = set(map(itemgetter(0), ev1[EVT.RELATED_TERMS]))
    all_terms_2 = set(map(itemgetter(0), ev2[EVT.RELATED_TERMS]))

    # Add the main terms to the sets of related terms
    # to get the final values of the sets of all terms.
    all_terms_1.update(main_terms_1)
    all_terms_2.update(main_terms_2)

    # Compute the similarity between the two events

    # text_similarity = st.szymkiewicz_simpson(main_terms_1, main_terms_2)
    text_similarity = st.szymkiewicz_simpson(all_terms_1, all_terms_2)
    time_similarity = st.overlap_coefficient(ev1[EVT.TIME_INTERVAL], ev2[EVT.TIME_INTERVAL])

    similarity = text_similarity * time_similarity
    assert 0 <= similarity <= 1

    return (1 - similarity)  # distance in [0, 1]


# https://stackoverflow.com/a/2573982
def symmetrize(a):
    """
    Return a symmetrized version of NumPy array a.

    Values 0 are replaced by the array value at the symmetric
    position (with respect to the diagonal), i.e. if a_ij = 0,
    then the returned array a' is such that a'_ij = a_ji.

    Diagonal values are left untouched.

    a -- square NumPy array, such that a_ij = 0 or a_ji = 0,
    for i != j.
    """
    return a + a.T - np.diag(a.diagonal())

def compute_distance_matrix(events):
    N = len(events)
    distance_matrix = np.zeros((N, N), dtype=np.double)

    for i, j in itertools.combinations(tqdm(range(N)), 2):
        ev1 = events[i]
        ev2 = events[j]
        dist = dist_event(ev1, ev2)
        distance_matrix[i, j] = dist

    distance_matrix = symmetrize(distance_matrix)

    return distance_matrix.tolist()


def fusion_cluster(events, labels):
    events_fusionned = []

    # retrouver les events par clusters
    labeled_events = dict()

    for i in range(len(events)):
        if labels[i] not in labeled_events:
            labeled_events[labels[i]] = []

        labeled_events[labels[i]].append(events[i])

    # afficher les clusters
    for label in sorted(labeled_events):
        print("Number of events in cluster {} : {}".format(label, len(labeled_events[label])))
        e = labeled_events[label]
        for x in e:
            print(f"\t{x[2]}")

    # fusionner les clusters
    for label in labeled_events:
        if label == -1:
            continue

        cluster = labeled_events[label]

        # trier les events dans les clusters par magnitude
        cluster.sort(key=lambda x: x[0], reverse=True)

        # debug
        # print("first and last of magnitude : {} {}".format(cluster[0][0], cluster[-1][0]))

        # fusionner les events dans les clusters
        # (mag, max_interval, vocabulary_entry[0], anomaly)
        mag = cluster[0][0]

        min_interval: int = 999999999
        max_interval: int = 0

        word_score = Counter()

        raw_anomaly = np.zeros_like(cluster[0][EVT.ANOMALY])
        for ev in cluster:
            ts, te = ev[EVT.TIME_INTERVAL]
            # trouver min et max interval
            if ts < min_interval:
                min_interval = ts
            if te > max_interval:
                max_interval = te

            # donner un score aux mots
            for word in ev[EVT.MAIN_TERM].split(', '):
                word_score.update({word: 1 + ev[0] / 10})

            # fusionner les anomalies
            raw_anomaly += ev[EVT.ANOMALY]

        # normaliser les anomalies
        raw_anomaly = list(raw_anomaly / len(cluster))

        main_term = ", ".join([w[0] for w in word_score.most_common(10)])

        related_terms = list(flatmap(lambda e: e[EVT.RELATED_TERMS], cluster))
        time_interval = (min_interval, max_interval)

        events_fusionned.append((mag, time_interval, main_term, related_terms, raw_anomaly))

    return events_fusionned


def run_with_auto_k(mabed: 'MABED', n_max: int = 100):
    #@mabed_cached(CacheLevel.L4_MABED, "distance_matrix")
    def get_distance_matrix(mabed: 'MABED'):
        return compute_distance_matrix(mabed.events)

    #@mabed_cached(CacheLevel.L4_MABED, "auto_merged_events")
    def cached_estimate(mabed: 'MABED'):
        distance_matrix = get_distance_matrix(mabed)

        clustering = cl.OPTICS(min_samples=2, max_eps=np.inf, metric="precomputed", n_jobs=-1).fit(distance_matrix)
        print(clustering.labels_)
        print("number of clusters : ", set(clustering.labels_))

        return fusion_cluster(mabed.events, clustering.labels_)

    # _prev_k = mabed.k
    mabed.k = n_max
    mabed.phase2()  # Detect events
    estimated_events = cached_estimate(mabed)
    # mabed.k = _prev_k

    mabed.event_graph = None
    mabed.redundancy_graph = None
    mabed.events = estimated_events

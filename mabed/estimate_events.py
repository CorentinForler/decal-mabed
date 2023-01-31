import itertools
from typing import TYPE_CHECKING
import numpy as np
from tqdm import tqdm
import sklearn.cluster as cl
from collections import Counter, defaultdict
from operator import itemgetter
from statistics import mean
import math

if TYPE_CHECKING:
    from mabed.mabed import MABED

from mabed.mabed_cache import CacheLevel, mabed_cached
import mabed.stats as st
from mabed.utils import EVT, get_main_and_related_terms, get_main_terms, get_related_terms


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def flatmap(func, *iterable):
    return itertools.chain.from_iterable(map(func, *iterable))


my_infinity = 1e99


def sigmoid(x):
    """
    Numerically-stable sigmoid function.
    https://stackoverflow.com/a/29863846
    """
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


def make_dist_function(aggregator: str = 'norm', properties: list = ['text', 'time', 'gap']):
    if '*' in properties:
        properties = ['text', 'time', 'gap']
    def dist_event__generated(ev1, ev2, ignored_terms=None):
        main_terms_1 = get_main_terms(ev1)
        main_terms_2 = get_main_terms(ev2)
        all_terms_1 = get_related_terms(ev1)
        all_terms_2 = get_related_terms(ev2)
        all_terms_1.update(main_terms_1)
        all_terms_2.update(main_terms_2)
        if ignored_terms:
            for s in (all_terms_1, main_terms_1, all_terms_2, main_terms_2):
                s.difference_update(ignored_terms)

        props = []
        if 'text' in properties:
            text_similarity = st.szymkiewicz_simpson(all_terms_1, all_terms_2)
            props.append(1 - text_similarity)
        t1, t2 = ev1[EVT.TIME_INTERVAL], ev2[EVT.TIME_INTERVAL]
        if 'time' in properties:
            overlap_coeff = st.overlap_coefficient(t1, t2)
            props.append(1 - overlap_coeff)
        if 'gap' in properties:
            time_distance = st.distance_in_time(t1, t2)
            props.append(max(0, time_distance))
        if 'erdem' in properties:
            imp1, imp2 = ev1[EVT.ANOMALY], ev2[EVT.ANOMALY]
            erdem_correl = st.erdem_correlation(imp1, imp2)
            props.append(sigmoid(-10.0 * erdem_correl))

        if 'norm' == aggregator:  # Euclidian norm
            return np.linalg.norm(props, ord=None) / math.sqrt(len(props))
        if 'sum' == aggregator:  # Sum
            return np.sum(props) / len(props)
        if 'prod' == aggregator:  # Product
            return np.prod(props)
        raise ArgumentError('aggregator not known')
    return dist_event__generated


def dist_event(ev1, ev2, ignored_terms=None):
    # Collect the main terms of the two events
    main_terms_1 = get_main_terms(ev1)
    main_terms_2 = get_main_terms(ev2)

    # Collect the related terms of the two events
    all_terms_1 = get_related_terms(ev1)
    all_terms_2 = get_related_terms(ev2)

    # Add the main terms to the sets of related terms
    # to get the final values of the sets of all terms.
    all_terms_1.update(main_terms_1)
    all_terms_2.update(main_terms_2)

    # Compute the similarity between the two events

    # 1. Compute the similarity based on the terms of the events
    if ignored_terms:
        for s in (all_terms_1, main_terms_1, all_terms_2, main_terms_2):
            s.difference_update(ignored_terms)

    # text_similarity = st.szymkiewicz_simpson(main_terms_1, main_terms_2)
    text_similarity = st.szymkiewicz_simpson(all_terms_1, all_terms_2)

    # 2. Compute the similarity based on the time-periods of the events
    imp1, imp2 = ev1[EVT.ANOMALY], ev2[EVT.ANOMALY]
    erdem_correl = st.erdem_correlation(imp1, imp2)
    erdem_correl = sigmoid(-10.0 * erdem_correl)

    distance = np.linalg.norm([1 - text_similarity, erdem_correl], ord=None) / math.sqrt(2)  # Euclidian norm
    # distance = max(0, distance - 0.5) / (1 - 0.5)
    # distance = min(distance / (1.0001 - distance), my_infinity)

    # print(main_terms_1, main_terms_2, '|', 'text_similarity:', text_similarity, '|', 'time_similarity:', time_similarity)
    # print('t1:', t1, 't2:', t2, 'overlap:', overlap_coeff, 'time_dist:', time_distance, 'time_close:', time_closeness)
    assert 0 <= distance <= 1
    return distance  # distance in [0, 1]


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

def compute_distance_matrix(events, dist_func=dist_event):
    N = len(events)
    distance_matrix = np.zeros((N, N), dtype=np.double)

    counter_for_terms = Counter()
    for ev in events:
        t = get_main_and_related_terms(ev)
        counter_for_terms.update(t)

    # rare_threshold = N // 2
    # rare_terms = { x for x, count in counter_for_terms.items() if count <= rare_threshold }
    # common_terms = { x for x, count in counter_for_terms.items() if count == N }
    # all_terms = set(counter_for_terms.keys())

    ignored_terms = set()
    # ignored_terms = common_terms # (all_terms - rare_terms) | common_terms
    # ignored_terms = (all_terms - rare_terms) | common_terms
    ignored_terms = frozenset(ignored_terms)

    # print('Stats:')
    # print('\x1b[34m', '- Common terms:', common_terms, '\x1b[0m')
    # print('\x1b[35m', '- Rare terms:', rare_terms, '\x1b[0m')
    # print('\x1b[2m',  '- All terms:', all_terms, '\x1b[0m')
    # print('\x1b[1m',  '- To ignore:', ignored_terms, '\x1b[0m')
    # input('Press <Enter> to continue.')

    for i, j in itertools.combinations(tqdm(range(N)), 2):
        ev1 = events[i]
        ev2 = events[j]
        dist = dist_func(ev1, ev2, ignored_terms=ignored_terms)
        # print(ev1[EVT.MAIN_TERM][:16], "-- vs --", ev2[EVT.MAIN_TERM][:16], dist)
        distance_matrix[i, j] = dist

    distance_matrix = symmetrize(distance_matrix)

    return distance_matrix.tolist()


def make_merged_clusters(events, labels, ignore_unclustered=True):
    merged_events = []

    # retrouver les events par clusters
    labeled_events = dict()

    for i in range(len(events)):
        if labels[i] not in labeled_events:
            labeled_events[labels[i]] = []

        labeled_events[labels[i]].append(events[i])

    # afficher les clusters
    e_faint = "\x1b[2m"
    e_reset = "\x1b[0m"
    def print_event(ev, depth=1):
        is_a_cluster = len(ev) > EVT.EXTRA and "cluster" in ev[EVT.EXTRA]
        if is_a_cluster:
            sub_events = ev[EVT.EXTRA]["cluster"]
            label = ev[EVT.EXTRA].get("cluster_label", "no-label")
            print(e_faint + ("│ " * depth) + "• Cluster [" + str(label) + "]:" + e_reset, ev[EVT.MAIN_TERM], e_faint + f"mag={ev[EVT.MAG]:.1f}" + e_reset)
            print(e_faint + ("│ " * depth) + f"├╴ {len(sub_events)} sub-events" + e_reset)
            for sub_ev in sorted(sub_events, key=itemgetter(EVT.MAG), reverse=True):
                print_event(sub_ev, depth=depth+1)
            print(e_faint + ("│ " * depth) + "└" + e_reset)
        else:
            print(e_faint + ("│ " * depth) + "• Event:" + e_reset, ev[EVT.MAIN_TERM], e_faint + f"mag={ev[EVT.MAG]:.1f}" + e_reset)

    # fusionner les clusters
    for label in labeled_events:
        if label == -1:
            if not ignore_unclustered:
                for ev in labeled_events[label]:
                    merged_events.append(ev)
            continue

        cluster = labeled_events[label]
        merged = merge_cluster(cluster)
        merged[EVT.EXTRA]["cluster_label"] = int(label)  # convert from int64
        merged_events.append(merged)

    merged_events.sort(key=itemgetter(EVT.MAG), reverse=True)
    for ev in merged_events:
        print_event(ev)

    return merged_events


def merge_cluster(cluster):
    # sort events in the cluster by magnitude
    cluster.sort(key=itemgetter(EVT.MAG), reverse=True)

    total_mag = 0
    min_interval: int = 999999999
    max_interval: int = 0

    raw_anomaly = np.zeros_like(cluster[0][EVT.ANOMALY])

    for ev in cluster:
        ts, te = ev[EVT.TIME_INTERVAL]
        # expand time interval of event
        if ts < min_interval:
            min_interval = ts
        if te > max_interval:
            max_interval = te

        # merge anomalies
        raw_anomaly += ev[EVT.ANOMALY]
        total_mag += ev[EVT.MAG]

    # normalize anomalies
    raw_anomaly = list(raw_anomaly / len(cluster))

    all_terms_raw = defaultdict(list)
    main_terms = set()
    for ev in cluster:
        ev_weight = ev[EVT.MAG] / total_mag  # normalized magnitude of the sub-event

        for t_term in ev[EVT.MAIN_TERM].split(', '):
            all_terms_raw[t_term].append((ev_weight, 1.0))

        for t_term, t_mag in ev[EVT.RELATED_TERMS]:
            all_terms_raw[t_term].append((ev_weight, t_mag))

    all_terms = Counter()
    for term, weighted_scores in all_terms_raw.items():
        all_terms[term] = sum([w * v for w, v in weighted_scores])
        all_terms[term] /= sum([w for w, _ in weighted_scores])

    # Normalize the terms values
    all_terms_max_count = max((count for _, count in all_terms.items()))
    all_terms = Counter({ key: count / all_terms_max_count for (key, count) in all_terms.items() })

    # Select main terms
    min_main_terms = max(2, min(len(all_terms) / 3,
        sum((ev[EVT.MAIN_TERM].count(',') + 1) for ev in cluster)
    ))
    main_terms_cutoff = 1.0
    n_main_terms = -1
    n_iter = 5
    while n_main_terms < min_main_terms and n_iter > 0:
        n_iter -= 1  # Prevent infinite loop
        n_main_terms = sum(1 for v in all_terms.values() if v >= main_terms_cutoff)
        main_terms_cutoff *= 0.95

    main_terms = all_terms.most_common(n_main_terms)
    main_terms_cutoff = min(map(itemgetter(1), main_terms))
    main_terms = set(map(itemgetter(0), main_terms))
    for term in main_terms:
        del all_terms[term]

    # Normalize related terms
    related_terms = []
    for term, score in all_terms.most_common():
        score /= main_terms_cutoff
        related_terms.append((term, score))

    all_terms = None

    # all_potential_main_terms = set(potential_main_terms.keys())
    # main_terms = set(w[0] for w in potential_main_terms.most_common(10))
    # skipped_main_terms = all_potential_main_terms - main_terms

    # related_terms_with_score = flatmap(lambda e: e[EVT.RELATED_TERMS], cluster)

    # related_terms = []
    # for tup in related_terms_with_score:
    #     word = tup[0]
    #     if word not in all_potential_main_terms:
    #         related_terms.append(tup)

    # for term in skipped_main_terms:
    #     related_terms.append((term, 1.0))

    # related_terms.sort(key=itemgetter(1), reverse=True)  # sort by score

    main_term = ", ".join(sorted(main_terms))
    time_interval = (min_interval, max_interval)
    extra = {"cluster": cluster}

    return (total_mag, time_interval, main_term, related_terms, raw_anomaly, extra)


def update_min_cluster_size(val):
    return val
    # n_events = 100
    # n_iter = 5
    # factor = 2
    # k = (math.log(n_events, factor)) / n_iter
    # mul = factor ** k
    # new_val = round(val * mul)
    # if new_val < 2:
    #     return 2
    # if new_val <= val:
    #     return val + 1
    # return new_val

def recursive_clustering(events, max_depth=0, label_offset=0, min_cluster_size=2, dist_func=dist_event):
    if min_cluster_size < 2:
        print("break: min_cluster_size <= 2")
        return [-1 for _ in events]

    if len(events) < min_cluster_size:
        print("break: len(events) <= min_cluster_size")
        return [-1 for _ in events]  # Minimum number of events

    distance_matrix = compute_distance_matrix(events, dist_func=dist_func)
    clustering = cl.OPTICS(min_samples=min_cluster_size, max_eps=np.inf, metric="precomputed", n_jobs=-1).fit(distance_matrix)
    labels = list(clustering.labels_)

    # print(min_cluster_size)
    # print(clustering.cluster_hierarchy_)
    print("Labels:", set(labels))

    # labels = [-1 for _ in events]
    # for ci, h in enumerate(clustering.cluster_hierarchy_):
    #     x, y = h
    #     for idx in range(x, y + 1):
    #         if labels[idx] <= -1:
    #             labels[idx] = ci


    # Offset the labels
    for i, label in enumerate(labels):
        if label != -1:
            labels[i] += label_offset

    n_clusters = len(set(labels))
    if n_clusters < 2 and max_depth > 0:
        print("break: not enough clusters")
        return [-1 for _ in events]  # Force recursion

    max_label = max(labels)
    if max_label < 0:
        max_label = -1

    # Recurse
    if max_depth > 0:
        map_indices = {}
        unclustered_events = []
        for idx, ev in enumerate(events):
            if labels[idx] == -1:
                new_idx = len(unclustered_events)
                unclustered_events.append(ev)
                map_indices[new_idx] = idx

        cluster_proxies = []
        for cluster_id in set(labels):
            if cluster_id < 0:
                continue

            sub_events = [ev for (idx, ev) in enumerate(events) if labels[idx] == cluster_id]
            merged = merge_cluster(sub_events)

            new_idx = len(unclustered_events) + len(cluster_proxies)
            cluster_proxies.append(merged)
            map_indices[new_idx] = cluster_id

        print(len(unclustered_events), 'unclustered events')
        print(len(cluster_proxies), 'cluster proxies')
        # ev1, ev2, *_ = unclustered_events
        # print('e.g.', ev1[EVT.MAIN_TERM][:16], "- vs -", ev2[EVT.MAIN_TERM][:16], dist_event(ev1, ev2))
        # input()
        new_labels = recursive_clustering(
            unclustered_events + cluster_proxies,
            max_depth=(max_depth - 1),
            label_offset=(max_label + 1),
            min_cluster_size=update_min_cluster_size(min_cluster_size),
            dist_func=dist_func,
        )

        for new_idx, new_label in enumerate(new_labels):
            idx = map_indices[new_idx]
            if new_label != -1:
                labels[idx] = new_label

    return labels


def run_with_auto_k(mabed: 'MABED', n_search: int = 100):
    mabed.k_orig = mabed.k
    mabed.k = n_search
    mabed.phase2()  # Detect events
    mabed.k = mabed.k_orig

    # don't try to force the clustering if k is 'auto'
    use_recursive_clustering = mabed.extra.get("recursive_clustering", mabed.k_orig > 0)

    # k == 0 means automatic k estimation, which means that we have to drop unclustered events
    drop_unclustered_events = mabed.extra.get("drop_unclustered_events", mabed.k_orig <= 0)

    #@mabed_cached(CacheLevel.L4_MABED, "auto_merged_events")
    def merge_events_by_clustering(mabed: 'MABED'):
        dist_func = dist_event

        if cluster_func := mabed.extra.get("cluster_func"):
            fname, fprops = cluster_func.split(":", 1)
            fname = fname.strip()
            fprops = list(map(lambda s: s.strip(), fprops.split(",")))
            dist_func = make_dist_function(fname, fprops)

        max_depth = 5 if use_recursive_clustering else 0
        labels = recursive_clustering(mabed.events, max_depth=max_depth, label_offset=0, min_cluster_size=2, dist_func=dist_func)
        print("Clusters:", set(labels))
        return make_merged_clusters(mabed.events, labels, ignore_unclustered=drop_unclustered_events)

    # event_filter = mabed.event_filter or noop_filter
    # filtered_events = event_filter(mabed.events, mabed)
    # if isinstance(filtered_events, list):
    #     print("info: event_filter returned a", type(filtered_events))
    #     filtered_events = list(filtered_events)
    # mabed.events = filtered_events or mabed.events

    if len(mabed.events) < 2:
        print(f"warning: not enough events ({len(mabed.events)}) to perform clustering.")
    else:
        # Only perform clustering if there are enough events
        mabed.events = merge_events_by_clustering(mabed)
        mabed.events = mabed.apply_filters(mabed.events, mabed.post_clustering_event_filters, 'post clustering filter')

    mabed.event_graph = None
    mabed.redundancy_graph = None


def noop_filter(events, mabed):
    return events


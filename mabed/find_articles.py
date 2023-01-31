
from collections import Counter
import itertools
import operator
from queue import PriorityQueue
from random import random

import numpy as np
from tqdm import tqdm

from mabed.mabed_cache import CacheLevel, cached_timeslice_read, mabed_cached


def get_sentence_vector(tup):
    nlp_doc = tup[3]
    return nlp_doc.vector  # TODO: compute a better vector representation


def find_articles_test_1(mabed, raw_events, n_articles):
    get_rich_event_from_raw = make_get_rich_event_from_raw(
        get_related_weight_method(None), n_articles
    )

    out_events, slices_to_inspect, map_words_to_events, all_events_representative_words = get_values_for_raw_events(
        get_rich_event_from_raw, raw_events)

    for i in tqdm(slices_to_inspect):
        for doc_text in cached_timeslice_read(mabed.corpus, i):
            words = set(mabed.corpus.tokenize_single_iterator(doc_text))
            for x in out_events:
                if i in x['slices']:
                    searched_words: set(str) = x['words']
                    priority_queue: PriorityQueue = x['best_articles']
                    score = len(words & searched_words)
                    if score > 1:
                        if priority_queue.full():
                            old = priority_queue.get()
                            if old[0] > score:
                                priority_queue.put(old)
                                continue
                        priority_queue.put((score, doc_text))
    out = []
    for x in out_events:
        prio_queue = x['best_articles']
        articles = []
        while not prio_queue.empty():
            articles.append(prio_queue.get())
        articles = sorted(articles, key=operator.itemgetter(0), reverse=True)
        # articles = list(map(operator.itemgetter(1), articles))
        out.append(articles)
    return out


def doc_iterator(mabed, slices: set, words_to_keep: set):
    for i in slices:
        for doc_text in cached_timeslice_read(mabed.corpus, i):
            if len(doc_text) > 400:
                continue
            tokenized = mabed.corpus.tokenize(doc_text)

            interesting_words = words_to_keep.intersection(tokenized)

            if len(interesting_words) > 0:
                yield tokenized, interesting_words, doc_text


def doc_iterator_parallel(mabed, slices: set, words_to_keep: set):
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool()

    def f(doc_text):
        tokenized = mabed.corpus.tokenize(doc_text)
        interesting_words = words_to_keep.intersection(tokenized)
        if len(interesting_words) > 0:
            return tokenized, interesting_words, doc_text

    it = iterate_mabed_slices(mabed, slices)
    return tqdm(filter(None, pool.imap_unordered(f, it)), desc=' -> iterating slices')


def nlp_iterator(it):
    print('Loading spacy...')
    import spacy
    nlp = spacy.load("en_core_web_lg")
    print('\x1b[ALoading spacy... done')

    for tup in it:
        yield *tup, nlp(tup[2])


def iterate_mabed_slices(mabed, slices):
    for i in slices:
        for doc_text in cached_timeslice_read(mabed.corpus, i):
            if len(doc_text) > 400:
                continue
            yield doc_text


def nlp_iterator_parallel(mabed, slices: set, words_to_keep: set):
    print('Loading spacy...')
    import spacy
    nlp = spacy.load("en_core_web_lg")
    print('\x1b[ALoading spacy... done')

    from multiprocessing.pool import ThreadPool
    pool = ThreadPool()

    def f(doc_text):
        tokenized = mabed.corpus.tokenize(doc_text)
        interesting_words = words_to_keep.intersection(tokenized)
        if len(interesting_words) > 0:
            return tokenized, interesting_words, doc_text, nlp(doc_text)

    it = iterate_mabed_slices(mabed, slices)
    return filter(None, tqdm(pool.imap_unordered(f, it), desc=' -> iterating slices'))


def get_ponderation_method(name: str):
    if name == 'count-ignore':
        return lambda count, weight: weight
    elif name == 'count-linear':
        return lambda count, weight: count * weight
    elif name == 'count-log':
        # return lambda count, weight: np.log1p(count) * weight
        return lambda count, weight: (1 + np.log(count)) * weight if count > 0 else 0
    else:
        return lambda count, weight: count * weight


def get_related_weight_method(base_weight: float = None):
    if base_weight is None:
        def rwm(mag_normalized, main_components):
            # return 0.9 * mag_normalized / (1 + np.log10(len(main_components)))
            return 0.9 / (1 + np.log10(len(main_components)))
            # return 0.9 * mag_normalized
            # return 0.5 * mag_normalized / len(main_components)
            # return 0.25 * mag_normalized
            # return 1 * mag_normalized
        return rwm
    else:
        def rwm(*x, **y):
            return base_weight
        return rwm


def make_get_rich_event_from_raw(compute_weight_for_related_term, pqueue_size: int):
    def f(event):
        main_term = event[2]
        main_components = main_term.split(', ')

        related_terms = event[3]
        related_terms = list(map(lambda x: x[0], related_terms))

        slice_start = event[1][0]
        slice_end = event[1][1]  # inclusive

        try:
            related_terms_max_weight = max((x[1] for x in event[3]))
        except:
            related_terms_max_weight = 1

        terms_weights = {}

        for word in main_components:
            terms_weights[word] = 1.0

        for term, mag in event[3]:
            weight = compute_weight_for_related_term(
                mag_normalized=mag / related_terms_max_weight,
                main_components=main_components,
            )
            terms_weights[term] = weight

        return {
            'event': event,
            'slices': list(range(slice_start, slice_end + 1)),
            'words': set(itertools.chain(main_components, related_terms)),
            'main_terms': set(main_components),
            'secondary_terms': set(related_terms),
            'best_articles': PriorityQueue(maxsize=pqueue_size),
            'best_article': None,
            'terms_weights': terms_weights,
        }
    return f


def simple_words_coverage(tup, event, **kwargs):
    a = tup[0]
    b = event['words']
    return len(b.intersection(a)) / len(b)


def get_scoring_method(name: str):
    if name == 'nlp':
        def scoring_method(tup, event, *, avg_proj, avg_proj_norm):
            nlp_doc = tup[3]
            score = np.dot(nlp_doc.vector, avg_proj)
            score /= (nlp_doc.vector_norm * avg_proj_norm)
            return score
        return scoring_method
    elif name == 'coverage-simple':
        return simple_words_coverage
    elif name.startswith('coverage-'):
        ponderation_method = get_ponderation_method(
            name[len('coverage-'):])

        def scoring_method(tup, event, **kwargs):
            return ponderated_words_coverage(ponderation_method, tup, event)
        return scoring_method


def ponderated_words_coverage(ponderation_method, tup, event):
    terms_weights = event['terms_weights']
    tokenized = tup[0]

    def f(t):
        term, count = t
        weight = terms_weights[term]
        return ponderation_method(count, weight)

    count = Counter(
        filter(terms_weights.__contains__, tokenized))
    score = sum(map(f, count.items()))

    return score


def get_values_for_raw_events(get_rich_event_from_raw, raw_events):
    out_events = list(map(get_rich_event_from_raw, raw_events))

    slices_to_inspect = set()
    all_events_representative_words = set()
    for event in out_events:
        slices_to_inspect.update(event['slices'])
        all_events_representative_words.update(event['words'])

    map_words_to_events = {}
    for w in all_events_representative_words:
        map_words_to_events[w] = []
        for event_index, event in enumerate(out_events):
            if w in event['words']:
                map_words_to_events[w].append(event_index)

    return out_events, slices_to_inspect, map_words_to_events, all_events_representative_words


def get_events_to_update_for_doc(map_words_to_events, tup):
    interesting_words = tup[1]

    events_to_update_for_this_document = set()

    for w in interesting_words:
        for event_index in map_words_to_events[w]:
            events_to_update_for_this_document.add(event_index)

    return events_to_update_for_this_document


def find_articles_for_events(
    mabed,
    raw_events,
    n_articles=3,
    secondary_term_fixed_weight=None,

    # 'nlp' or 'coverage-simple' or 'coverage-count-ignore' or 'coverage-count-linear' or 'coverage-count-log'
    scoring_method_name='coverage-count-log',
    divide_score_by_length=False,
    use_nlp=False,
):
    assert n_articles >= 1, "find_articles_for_events: n_articles should be at least 1"

    compute_weight_for_related_term = get_related_weight_method(
        secondary_term_fixed_weight)

    scoring_method = get_scoring_method(scoring_method_name)

    @mabed_cached(CacheLevel.L4_MABED, "articles", extra_data={'n_articles': n_articles, 'scoring_method_name': scoring_method_name, 'secondary_term_fixed_weight': secondary_term_fixed_weight})
    def func(mabed):
        assert len(raw_events) > 0

        get_rich_event_from_raw = make_get_rich_event_from_raw(
            compute_weight_for_related_term, n_articles)
        out_events, slices_to_inspect, map_words_to_events, all_events_representative_words = get_values_for_raw_events(
            get_rich_event_from_raw, raw_events)

        tuples_by_event = {}
        for event_index, event in enumerate(out_events):
            tuples_by_event[event_index] = []

        if use_nlp:
            # Calcul des titres représentatifs avec :
            #   - fonction de distance prenant en compte le nombre d'occurrences de termes
            #   - par projection dans un espace vectoriel

            it = nlp_iterator_parallel(
                mabed=mabed,
                slices=slices_to_inspect,
                words_to_keep=all_events_representative_words)
        else:
            it = doc_iterator_parallel(
                mabed=mabed,
                slices=slices_to_inspect,
                words_to_keep=all_events_representative_words)

        it = tqdm(it, desc="Computing projections")
        for tup in it:
            for event_index in get_events_to_update_for_doc(map_words_to_events, tup):
                tuples_by_event[event_index].append(tup)

        it = enumerate(out_events)
        it = tqdm(it, total=len(out_events), desc='Finding best article')
        for event_index, event in it:
            tuples = tuples_by_event[event_index]

            avg_proj = None
            avg_proj_norm = None
            if use_nlp:
                all_terms = event['words']
                # thresh = int(max(1, len(all_terms) * 0.25))

                # compute sentence vector representation for the text
                # https://spacy.io/usage/vectors-similarity

                def g(t):
                    return get_sentence_vector(t)
                    # return simple_words_coverage(t, event) * get_sentence_vector(t)

                def f(t):
                    interesting_words = all_terms.intersection(t[1])
                    return len(interesting_words) >= 1
                    # return len(interesting_words) >= thresh

                projections = list(map(g, filter(f, tuples)))
                if len(projections) == 0:
                    print(
                        f'\x1b[31mNo projection found for event {event_index}\x1b[0m')
                    continue
                avg_proj = np.mean(projections, axis=0)
                avg_proj_norm = np.linalg.norm(avg_proj)

            for tup in tuples:
                # TODO: rajouter un filtrage ?
                # TF-IDF? -> clustering des titres
                # trouver le titre, représentatif d'un événement/d'un ensemble de textes,
                # le plus fréquent à quelques mots près
                # --> prendre un représentant d'un cluster

                score = scoring_method(
                    tup=tup,
                    event=event,
                    avg_proj=avg_proj,
                    avg_proj_norm=avg_proj_norm,
                )

                if divide_score_by_length:
                    # score /= len(tup[0])  # divide by number of words
                    w = len(tup[0])
                    score /= np.log10(1 + 0.2 * w)

                pqueue_push(event['best_articles'], score, tup[2])

        return list(map(pqueue_get_all, map(operator.itemgetter('best_articles'), out_events)))

    return func(mabed)


def describe_events_okapi_bm25(mabed, raw_events, n_articles=5):
    """
    Find articles for a list of raw events using Okapi BM25.
    """
    from rank_bm25 import BM25Okapi

    compute_weight_for_related_term = get_related_weight_method(0.5)

    get_rich_event_from_raw = make_get_rich_event_from_raw(
        compute_weight_for_related_term, n_articles)
    out_events, slices_to_inspect, map_words_to_events, all_events_representative_words = get_values_for_raw_events(
        get_rich_event_from_raw, raw_events)

    tuples_by_event = []
    for event_index, event in enumerate(out_events):
        tuples_by_event.append([])

    it = doc_iterator(
        mabed=mabed,
        slices=slices_to_inspect,
        words_to_keep=all_events_representative_words)

    it = tqdm(it, desc="Iterating articles")
    for tup in it:
        for event_index in get_events_to_update_for_doc(map_words_to_events, tup):
            tuples_by_event[event_index].append(tup)

    it = zip(out_events, tuples_by_event)
    it = tqdm(it, total=len(out_events),
              desc='Finding best article(s) for events')
    for (event, tuples) in it:
        tokenized_corpus = map(operator.itemgetter(0), tuples)
        bm25 = BM25Okapi(tokenized_corpus)

        query = list(event['words'])
        scores = bm25.get_scores(query)

        for score, tup in zip(scores, tuples):
            pqueue_push(event['best_articles'], score, tup[2])

    return list(map(pqueue_get_all, map(operator.itemgetter('best_articles'), out_events)))


def pqueue_push(priority_queue: PriorityQueue, score: float, item: tuple):
    if priority_queue.full():
        old = priority_queue.get()
        if old and old[0] > score:
            priority_queue.put(old)
            return
    priority_queue.put((score, item))


def pqueue_get_all(priority_queue: PriorityQueue):
    out = []
    while not priority_queue.empty():
        out.append(priority_queue.get())
    out = sorted(out, key=operator.itemgetter(0), reverse=True)
    # out = list(map(operator.itemgetter(1), out))
    return out


def iterate_all_articles_for_periods(mabed, periods: list):
    all_articles = mabed.corpus.source_csv_iterator()
    for date, text in tqdm(all_articles, desc='Iterating articles'):
        # for each article, find which period(s) it belongs to
        # (and these periods are linked to events)
        periodIndices = set()

        for periodIndex, period in enumerate(periods):
            ts, te = period
            if ts <= date <= te:
                periodIndices.add(periodIndex)

        if periodIndices:
            yield periodIndices, date, text

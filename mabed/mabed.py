# coding: utf-8

# std
from collections import Counter
import itertools
from multiprocessing.dummy import Pool as ThreadPool
from operator import itemgetter
import operator
import os
from queue import PriorityQueue

# math
import networkx as nx
import numpy as np
from mabed.estimate_events import run_with_auto_k
from mabed.find_articles import find_articles_for_events, iterate_all_articles_for_periods
from mabed.mabed_cache import JSON_EXTENSION, PICKLE_EXTENSION, CacheLevel, cached_timeslice_read, mabed_cached
import mabed.stats as st
import mabed.utils as utils
from tqdm import tqdm

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mabed.corpus import Corpus

__authors__ = "Adrien Guille, Nicolas Dugué"
__email__ = "adrien.guille@univ-lyon2.fr"


class MABED:

    def __init__(self, corpus: 'Corpus', extra: dict = None):
        self.corpus = corpus
        self.event_graph = None
        self.redundancy_graph = None
        self.events = None
        self.p = None
        self.k = None
        self.theta = None
        self.sigma = None
        self.extra = extra or dict()

    def run(self, k: int = 10, p: int = 10, theta: float = 0.6, sigma: float = 0.5, event_filter=None):
        self.p = p
        self.k = k
        self.theta = theta
        self.sigma = sigma

        self.event_filters = []
        self.post_clustering_event_filters = []
        if event_filter:
            self.event_filters.append(event_filter)

        if not isinstance(self.k, int) or self.k <= 0:
            if isinstance(self.k, int) and self.k < 0:
                # HACK: If k is negative, it's a hint that we have to
                # truncate the list of events. Set up the filter
                # by chaining it with self.event_filter
                k = -self.k
                def truncator(events, mabed):
                    return events[:k]

                self.post_clustering_event_filters.append(truncator)

            n_search = 100  # max events with auto k
            return run_with_auto_k(self, n_search=n_search)

        # basic_events = self.phase1()
        basic_events = None  # may use phase2 cache without loading phase1
        return self.phase2(basic_events)

    @mabed_cached(CacheLevel.L3_DISCRETE, 'basic_events', JSON_EXTENSION)
    def phase1(self):
        print('Phase 1...')

        items = self.corpus.vocabulary.items()
        n = len(items)

        n_processors = os.cpu_count() or 1
        pool = ThreadPool(n_processors)

        async_results = pool.imap_unordered(
            func=self.maximum_contiguous_subsequence_sum,
            iterable=items
        )

        basic_events = tqdm(
            async_results,
            desc="computing basic events",
            total=n)

        # for vocabulary_entry in tqdm(self.corpus.vocabulary.items(), desc="starting multi-threading", total=n):
        #     async_results.append(pool.apply_async(
        #         self.maximum_contiguous_subsequence_sum, (vocabulary_entry,)))

        # show progress
        # for e in tqdm(async_results, desc="computing basic events", total=n):
        #     basic_events.append(e.get())

        basic_events = sorted(
            basic_events, key=itemgetter(0), reverse=True)

        pool.close()
        pool.join()

        # for vocabulary_entry in self.corpus.vocabulary.items():
        #     basic_events.append(
        #         self.maximum_contiguous_subsequence_sum(vocabulary_entry))
        print('   Detected events: %d' % len(basic_events))
        return basic_events

    def maximum_contiguous_subsequence_sum(self, vocabulary_entry):
        mention_freq = self.corpus.mention_freq[vocabulary_entry[1], :].toarray(
        )
        mention_freq = mention_freq[0, :]
        total_mention_freq = np.sum(mention_freq)

        # compute the time-series that describes the evolution of mention-anomaly
        anomaly = []
        for i in range(0, self.corpus.time_slice_count):
            anomaly.append(self.anomaly(
                i, mention_freq[i], total_mention_freq))
        max_ending_here = max_so_far = 0
        a = b = a_ending_here = 0
        for idx, ano in enumerate(anomaly):
            max_ending_here = max(0, max_ending_here + ano)
            if max_ending_here == 0:
                # a new bigger sum may start from here
                a_ending_here = idx
            if max_ending_here > max_so_far:
                # the new sum from a_ending_here to idx is bigger
                a = a_ending_here+1
                max_so_far = max_ending_here
                b = idx

        # return the event description
        max_interval = (a, b)
        mag = np.sum(anomaly[a:b+1])
        basic_event = (mag, max_interval, vocabulary_entry[0], anomaly)
        return basic_event

    def phase2(self, basic_events=None):
        o = self.compute_phase2(basic_events)
        self.event_graph = o['event_graph']
        self.redundancy_graph = o['redundancy_graph']
        self.events = o['events']
        self.perform_filtering('post phase 2 (maybe cached)')

    def perform_filtering(self, label='- perform_filtering'):
        self.events = self.apply_filters(self.events, self.event_filters, label)

    def apply_filters(self, events=None, filters=None, label=''):
        events = events or self.events
        filters = filters or self.event_filters

        if filters and events:
            print('┌╴ Filter stats', label)
            print('│  #filters:', len(filters))
            print('│  input:', len(events), 'events')
            for i, filt in enumerate(filters):
                print(f'├╴ after filter #{i}: ', end='')
                o = filt(events, self)
                is_idempotent = o == filt(events, self)
                if not is_idempotent:
                    raise Exception('not idempotent')
                events = list(o) or events
                print(f'{len(events)} events')
            print('└╴ final:', len(events), 'events')
            print()
            # input('Press <Enter> to continue.')

        return events

    @mabed_cached(CacheLevel.L4_MABED, 'mabed_phase_2', PICKLE_EXTENSION)
    def compute_phase2(self, basic_events):
        if basic_events is None:
            basic_events = self.phase1()

        print('Phase 2...')

        # sort the events detected during phase 1 according to their magnitude of impact
        basic_events.sort(key=itemgetter(0), reverse=True)

        # create the event graph (directed) and the redundancy graph (undirected)
        self.event_graph = nx.DiGraph(name='Event graph')
        self.redundancy_graph = nx.Graph(name='Redundancy graph')
        i = 0
        unique_events = 0
        refined_events = []

        # phase 2 goes on until the top k (distinct) events have been identified
        while unique_events < self.k and i < len(basic_events):
            basic_event = basic_events[i]
            main_word = basic_event[2]
            candidate_words = self.corpus.cooccurring_words(
                basic_event, self.p)
            main_word_freq = self.corpus.global_freq[self.corpus.vocabulary[main_word], :].toarray(
            )
            main_word_freq = main_word_freq[0, :]
            related_words = []

            # identify candidate words based on co-occurrence
            if candidate_words is not None:
                for candidate_word in candidate_words:
                    candidate_word_freq = self.corpus.global_freq[self.corpus.vocabulary[candidate_word], :].toarray(
                    )
                    candidate_word_freq = candidate_word_freq[0, :]

                    # compute correlation and filter according to theta
                    # weight = (st.erdem_correlation(
                    #     main_word_freq, candidate_word_freq) + 1) / 2
                    weight = st.erdem_correlation_java(
                        main_word_freq, candidate_word_freq, basic_event[1][0], basic_event[1][1])
                    if weight >= self.theta:
                        related_words.append((candidate_word, weight))

                if len(related_words) > 1:
                    refined_event = (
                        basic_event[0], basic_event[1], main_word, related_words, basic_event[3])
                    # check if this event is distinct from those already stored in the event graph
                    if self.update_graphs(refined_event):
                        refined_events.append(refined_event)
                        unique_events += 1
            i += 1

        self.perform_filtering('inside phase 2 computation (no cache)')

        # merge redundant events and save the result
        self.events = self.merge_redundant_events(refined_events)

        return {
            'event_graph': self.event_graph,
            'redundancy_graph': self.redundancy_graph,
            'events': self.events,
        }

    def anomaly(self, time_slice, observation, total_mention_freq):
        # compute the expected frequency of the given word at this time-slice
        expectation = float(self.corpus.tweet_count[time_slice]) * (
            float(total_mention_freq)/(float(self.corpus.size)))

        # return the difference between the observed frequency and the expected frequency
        return observation - expectation

    def update_graphs(self, event):
        redundant = False
        main_word = event[2]
        # check whether 'event' is redundant with another event already stored in the event graph or not
        if self.event_graph.has_node(main_word):
            for related_word, weight in event[3]:
                if self.event_graph.has_edge(main_word, related_word):
                    interval_0 = self.event_graph.nodes[related_word]['interval']
                    interval_1 = event[1]
                    if st.overlap_coefficient(interval_0, interval_1) > self.sigma:
                        self.redundancy_graph.add_node(
                            main_word, description=event)
                        self.redundancy_graph.add_node(
                            related_word, description=self.get_event(related_word))
                        self.redundancy_graph.add_edge(main_word, related_word)
                        redundant = True
                        break
        if not redundant:
            self.event_graph.add_node(
                event[2], interval=event[1], mag=event[0], main_term=True)
            for related_word, weight in event[3]:
                self.event_graph.add_edge(
                    related_word, event[2], weight=weight)
        return not redundant

    def get_event(self, main_term):
        if self.event_graph.has_node(main_term):
            event_node = self.event_graph.nodes[main_term]
            if event_node['main_term']:
                related_words = []
                for node in self.event_graph.predecessors(main_term):
                    related_words.append(
                        (node, self.event_graph.get_edge_data(node, main_term)['weight']))
                return event_node['mag'], event_node['interval'], main_term, related_words

    def merge_redundant_events(self, events):
        # compute the connected components in the redundancy graph
        components = []
        for c in nx.connected_components(self.redundancy_graph):
            components.append(c)
        final_events = []

        # merge redundant events
        for event in events:
            main_word = event[2]
            main_term = main_word
            descriptions = []
            for component in components:
                if main_word in component:
                    main_term = ', '.join(component)
                    for node in component:
                        descriptions.append(
                            self.redundancy_graph.nodes[node]['description'])
                    break
            if len(descriptions) == 0:
                related_words = event[3]
            else:
                related_words = self.merge_related_words(
                    main_term, descriptions)
            final_event = (event[0], event[1], main_term,
                           related_words, event[4])
            final_events.append(final_event)
        return final_events

    def merge_related_words(self, main_term, descriptions):
        all_related_words = []
        for desc in descriptions:
            all_related_words.extend(desc[3])
        all_related_words.sort(key=lambda tup: tup[1], reverse=True)
        merged_related_words = []
        for word, weight in all_related_words:
            if word not in main_term and dict(merged_related_words).get(word) is None:
                if len(merged_related_words) == self.p:
                    break
                merged_related_words.append((word, weight))
        return merged_related_words

    def print_event(self, event):
        related_words = []
        for related_word, weight in sorted(event[3], key=1):
            related_words.append(f'{related_word} ({100*weight:.1f}%)')

        dt_beg = str(self.corpus.to_date(event[1][0]))
        dt_end = str(self.corpus.to_date(event[1][1]))

        print(f'┌╴ {event[2]}')
        print(f'│ mag:   {event[0]}')
        print(f'│ start: {dt_beg}')
        print(f'│ end:   {dt_end}')
        print(f'│ related: {", ".join(related_words)}')
        print(f'└╴')

    def print_events(self):
        print('   Top %d events:' % len(self.events))
        print()

        for event in self.events:
            self.print_event(event)

    def find_articles_for_events(self, n_articles=1, **kwargs):
        assert n_articles >= 1, "mabed.find_articles_for_events: n_articles should be at least 1"
        return find_articles_for_events(mabed=self, n_articles=n_articles, raw_events=self.events, **kwargs)

    # @mabed_cached(CacheLevel.L4_MABED, 'cytoscape_graph', JSON_EXTENSION)
    def as_cytoscape_graph(self):
        events = list(utils.iterate_events_as_dict(self))

        out = []
        idsToIndex = {}

        for idx, e in tqdm(enumerate(events), unit=' events', desc='Creating nodes from events'):
            idsToIndex[id] = idx
            mainTerms = e['term'].split(', ')

            out.append({
                "group": "nodes",
                "data": {
                    "label": ', '.join(mainTerms),
                    "dateStart": e['start'],
                    "dateEnd": e['end'],
                    "id": idx,
                },
            })
            idx += 1

        def rangeTriangle(n):
            for i in range(n):
                for j in range(i + 1, n):
                    yield i, j

        pairs = rangeTriangle(len(out))
        edgesToCompute: list = []
        edgeTerms: list = []
        for i, j in tqdm(pairs, unit=' pairs', desc='Creating edges between events'):
            sourceId = i  # out[i]['data']['id']
            targetId = j  # out[j]['data']['id']
            sourceEvent = events[i]
            targetEvent = events[j]

            dateStart = max(sourceEvent['start'], targetEvent['start'])
            dateEnd = min(sourceEvent['end'], targetEvent['end'])

            edge = {
                "group": "edges",
                "data": {
                    "source": sourceId,
                    "target": targetId,
                    "weight": 0,
                    "dateStart": dateStart,
                    "dateEnd": dateEnd,
                    "id": f'{sourceId}-{targetId}',
                },
            }
            edgesToCompute.append(edge)
            sourceTerms = utils.getAllTermsForEvent(sourceEvent)
            targetTerms = utils.getAllTermsForEvent(targetEvent)
            edgeTerms.append((sourceTerms, targetTerms))
            out.append(edge)

        def update_weights(tup):
            edgeIndices, date, text = tup
            for edgeIndex in edgeIndices:
                edge = edgesToCompute[edgeIndex]
                sourceTerms, targetTerms = edgeTerms[edgeIndex]

                if date < edge['data']['dateStart'] or date > edge['data']['dateEnd']:
                    continue

                sourceTermsInText = [t for t in sourceTerms if t in text]
                targetTermsInText = [t for t in targetTerms if t in text]

                if len(sourceTermsInText) == 0 or len(targetTermsInText) == 0:
                    continue

                edge['data']['weight'] += 1  # len(sourceTermsInText) * len(targetTermsInText)

        def compute_all_weights():
            periods = [ (e['data']['dateStart'], e['data']['dateEnd']) for e in edgesToCompute ]
            it = iterate_all_articles_for_periods(self, periods)
            p = ThreadPool(32)
            it = p.imap_unordered(update_weights, it)
            it = tqdm(it, total=len(periods), unit=' periods', desc='Computing edge weights')
            p.close()
            list(it) # consume iterator

        def set_all_weights_to_one():
            for e in edgesToCompute:
                e['data']['weight'] = 1

        compute_all_weights()
        # set_all_weights_to_one()

        out = list(filter(lambda e: e['group'] != 'edges' or e['data']['weight'] > 0, out))
        maxWeight = max([e['data']['weight'] for e in out if e['group'] == 'edges'])
        for e in out:
            if e['group'] == 'edges':
                w = e['data']['weight']
                e['data']['rawWeight'] = w
                e['data']['weight'] = w / maxWeight

        return out

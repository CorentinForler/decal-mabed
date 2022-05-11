import collections
import json
import os
import time
from datetime import datetime

import numpy as np
import sklearn.cluster as cl
from tqdm import tqdm

import mabed.stats as st
from mabed.corpus import Corpus
from mabed.mabed import MABED

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

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

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

def fusion_cluster(basic_events, labels):
    events_fusionned = []
    
    # retrouver les basic_events par clusters
    labeled_basic_events = dict()

    for i in range(len(basic_events)):
        if labels[i] not in labeled_basic_events:
            labeled_basic_events[labels[i]] = []

        labeled_basic_events[labels[i]].append(basic_events[i])

    # afficher les clusters
    for label in labeled_basic_events:
        print("Number of events in cluster {} : {}".format(label, len(labeled_basic_events[label])))

    # fusionner les clusters
    for label in labeled_basic_events:

        if label == -1:
            continue

        # trier les basic_events dans les clusters par magnitude
        labeled_basic_events[label].sort(key=lambda x: x[0], reverse=True)

        # debug
        # print("first and last of magnitude : {} {}".format(labeled_basic_events[label][0][0], labeled_basic_events[label][-1][0]))

        # fusionner les basic_events dans les clusters
        # (mag, max_interval, vocabulary_entry[0], anomaly)
        mag = labeled_basic_events[label][0][0]
        
        min_interval = 999999999
        max_interval = 0

        word_score = collections.Counter()

        for ev in labeled_basic_events[label]:

            # trouver min et max interval
            if ev[1][0] < min_interval:
                min_interval = ev[1][0]
            if ev[1][1] > max_interval:
                max_interval = ev[1][1]

            # donner un score aux mots
            for word in ev[2].split(', '):
                word_score.update({word: 1 + ev[0]/10})
        
        main_words = ", ".join([x[0] for x in word_score.most_common(10)])
        # print(main_words)

        events_fusionned.append((mag, (min_interval, max_interval), main_words, labeled_basic_events[label][0][3]))
    
    return events_fusionned
        

    


if __name__ == '__main__':
    start_time = time.time()
    
    mabed = get_mabed()

    # Paramètres
    Nb_ev = 1000 # Nombre d'events que MABED doit trouver
    p = 10
    theta = 0.6
    sigma = 0.5

    mabed.run(Nb_ev, p, theta, sigma)

    compute_matrix = time.time()
    distance_matrix_path = "distance_matrix_" + str(Nb_ev) +"_" + str(p) +"_" + str(theta) + "_" + str(sigma) + ".json"


    if os.path.exists(distance_matrix_path):
        with open(distance_matrix_path, "r") as f:
            distance_matrix = json.load(f)
    else:
        distance_matrix = None

    if distance_matrix is None:
        distance_matrix = compute_distance_matrix(mabed.events, mabed)

        with open(distance_matrix_path, "w") as file_dm:
            json.dump(distance_matrix.tolist(), file_dm)
    end_compute_matrix = time.time()


    clustering_time = time.time()
    clustering = cl.OPTICS(min_samples=2, max_eps=np.inf, metric="precomputed", n_jobs=-1).fit(distance_matrix)

    
    end_clustering_time = time.time()
    print(clustering.labels_)
    print("number of clusters : ", set(clustering.labels_))
    
    events = fusion_cluster(mabed.events, clustering.labels_)

    print("number of events : ", len(events))

    end_time = time.time()
    print("basic_event :", len(distance_matrix[0]))
    print("duration:", end_time - start_time)
    print("compute_matrix : ", end_compute_matrix - compute_matrix)
    print("clustering time : ", end_clustering_time - clustering_time)

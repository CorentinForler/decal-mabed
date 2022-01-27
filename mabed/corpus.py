# coding: utf-8

# std
import string
from datetime import timedelta, datetime
import csv
import os
import shutil
import operator

# math
import numpy as np
# from scipy.sparse import *
from scipy.sparse.dok import dok_matrix
from mabed.mabed_cache import CacheLevel, cached_getpath, cached_timeslice_append, cached_timeslice_init, cached_timeslice_read, cached_timeslices, corpus_cached

# mabed
import mabed.utils as utils

__authors__ = "Adrien Guille, Nicolas Dugué"
__email__ = "adrien.guille@univ-lyon2.fr"

# DATETIME_FORMAT="%Y-%m-%d %H:%M:%S"
DATETIME_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT_LENGTH = len(DATETIME_FORMAT.replace('%Y', '1234'))


class Corpus:
    def __init__(self, source_file_path, stopwords_file_path, min_absolute_freq=10, max_relative_freq=0.4, separator='\t', save_voc=False):
        self.source_file_path = source_file_path
        self.separator = separator
        self.min_absolute_freq = min_absolute_freq
        self.max_relative_freq = max_relative_freq

        self.size = 0

        # discretization params
        self.time_slice_count = None
        self.tweet_count = None
        self.global_freq = None
        self.mention_freq = None
        self.time_slice_length = None

        self.start_date = '3000-01-01 00:00:00'[:DATETIME_FORMAT_LENGTH]
        self.end_date = '1970-01-01 00:00:00'[:DATETIME_FORMAT_LENGTH]
        self.min_date = '1970-01-01 00:00:00'[:DATETIME_FORMAT_LENGTH]

        # load stop-words
        self.stopwords = utils.load_stopwords(stopwords_file_path)

        vocab_vector, size, date_start, date_end = self.compute_vocabulary_vector()

        assert(size > 0)
        assert(date_start <= date_end)
        assert(len(vocab_vector) > 0)

        self.size = size
        self.start_date = datetime.strptime(date_start, DATETIME_FORMAT)
        self.end_date = datetime.strptime(date_end, DATETIME_FORMAT)

        if save_voc:
            utils.write_vocabulary(vocab_vector)

        # construct the vocabulary map
        self.vocabulary = self.compute_filtered_vocabulary_map(
            min_absolute_freq, max_relative_freq, vocab_vector)
        print('   Filtered vocabulary: %d distinct words' %
              len(self.vocabulary))

        print('   Corpus: %i tweets, spanning from %s to %s' %
              (self.size, self.start_date, self.end_date))

    @corpus_cached(CacheLevel.L2_VOCAB, "vocab_map")
    def compute_filtered_vocabulary_map(self, min_absolute_freq, max_relative_freq, vocab_vector):
        vocab_map = {}
        word_index = 0
        for word, frequency in vocab_vector:
            if frequency > min_absolute_freq and float(frequency / self.size) < max_relative_freq and word not in self.stopwords:
                vocab_map[word] = word_index
                word_index += 1
        return vocab_map

    @corpus_cached(CacheLevel.L1_DATASET, "vocab_vector")
    def compute_vocabulary_vector(self):
        date_start = '3000-01-01 00:00:00'[:DATETIME_FORMAT_LENGTH]
        date_end = '1970-01-01 00:00:00'[:DATETIME_FORMAT_LENGTH]
        size = 0
        word_frequency = {}

        for (date, text) in self.source_csv_iterator():
            size += 1
            words = self.tokenize(text)

            if date > date_end:
                date_end = date
            elif date < date_start:
                date_start = date

            # update word frequency
            for word in words:
                if word not in self.stopwords:
                    if len(word) > 1:
                        if word not in word_frequency:
                            word_frequency[word] = 0
                        word_frequency[word] += 1

        # sort words w.r.t frequency
        vocabulary = list(word_frequency.items())
        vocabulary.sort(key=operator.itemgetter(1), reverse=True)
        return vocabulary, size, date_start, date_end

    def source_csv_iterator(self):
        with open(self.source_file_path, 'r', encoding='utf8') as input_file:
            csv_reader = csv.reader(input_file, delimiter=self.separator)
            header = next(csv_reader)
            text_column_index = header.index('text')
            date_column_index = header.index('date')

            for line in csv_reader:
                # if len(line) != 4:
                #     print('skipping line:', line)
                #     continue
                date = line[date_column_index]
                text = line[text_column_index]

                if date < self.min_date:
                    print('skipping line:', line)
                    continue  # ignore

                if not text:
                    continue

                yield date, text

    def tokenized_iterator(self):
        path = cached_getpath(self, CacheLevel.L1_DATASET, "tokenized", ".csv")

        # import spacy
        # nlp = spacy.load("en_core_web_sm")

        if os.path.exists(path):
            with open(path, 'r', encoding='utf8') as input_file:
                csv_reader = csv.reader(input_file, delimiter='\t')
                for line in csv_reader:
                    # date, mention, text, tokens
                    yield line[0], line[1], line[2], line[3:]
        else:
            with open(path, 'w', encoding='utf8') as output_file:
                csv_writer = csv.writer(output_file, delimiter='\t')
                for (date, text) in self.source_csv_iterator():

                    # tokenize the tweet and update word frequency
                    words = self.tokenize(text)

                    # mention = '@' in text
                    mention = 'Apple' in text
                    # nlp_text = nlp(text)
                    # orgs = filter(lambda t: t.ent_type_ ==
                    #               'ORG' and len(t.text) > 1, nlp_text)
                    # has_orgs = any(orgs) and any(orgs)
                    # mention = has_orgs

                    csv_writer.writerow([date, mention, text, *words])
                    yield (date, mention, text, words)

    # def import_discretized(self, file_path, time_slice_length):
    #     if os.path.exists(file_path):
    #         print(f'Importing previous corpus from {file_path} and corpus/')
    #         try:
    #             data = utils.load_pickle(file_path)
    #             self.time_slice_length = data['time_slice_length']
    #             self.time_slice_count = data['time_slice_count']
    #             self.tweet_count = data['tweet_count']
    #             self.global_freq = data['global_freq']
    #             self.mention_freq = data['mention_freq']

    #             if data['source_file_path'] != self.source_file_path:
    #                 raise Exception('Warning: .source_file_path mismatch')
    #             if data['size'] != self.size:
    #                 raise Exception('Warning: .size mismatch')
    #             if data['start_date'] != self.start_date:
    #                 raise Exception('Warning: .start_date mismatch')
    #             if data['end_date'] != self.end_date:
    #                 raise Exception('Warning: .end_date mismatch')
    #             if data['min_date'] != self.min_date:
    #                 raise Exception('Warning: .min_date mismatch')

    #             if time_slice_length != self.time_slice_length:
    #                 print('Warning: stored corpus has different time_slice_length')
    #                 raise Exception(
    #                     f'Expected: time_slice_length = {time_slice_length}, Got: {self.time_slice_length}')

    #             for i in range(self.time_slice_count):
    #                 if not utils.time_slice_exists(i):
    #                     raise Exception(f'Missing corpus time slice file: {i}')

    #             return True

    #         except Exception as e:
    #             print('Error while importing discretized corpus')
    #             print(e)

    #     print('Doing discretization now')
    #     self.discretize(time_slice_length)
    #     self.save_discretized(file_path)
    #     return False

    # def save_discretized(self, file_path):
    #     data = utils.pick_fields(self, [
    #         'time_slice_count', 'tweet_count', 'global_freq', 'mention_freq', 'time_slice_length',
    #         'source_file_path', 'size', 'start_date', 'end_date', 'min_date'])
    #     utils.save_pickle(data, file_path)

    def compute_tokenized_corpus(self):
        self.tokenized_corpus = []
        for (date, text) in self.source_csv_iterator():
            self.tokenized_corpus.append((date, self.tokenize(text)))

    @corpus_cached(CacheLevel.L3_DISCRETE, "discretized_corpus")
    def compute_discretized_corpus(self):
        time_slice_length = self.time_slice_length
        start_date, end_date = self.start_date, self.end_date
        vocab = self.vocabulary

        # clean the data directory
        if os.path.exists('corpus'):
            shutil.rmtree('corpus')
        os.makedirs('corpus')

        def get_time_slice_index(date):
            time_delta = (date - start_date)
            time_delta = time_delta.total_seconds() / 60
            return int(time_delta // time_slice_length)

        # compute the total number of time-slices
        time_slice_count = get_time_slice_index(end_date) + 1
        tweet_count = np.zeros(time_slice_count)
        print(' Number of time-slices: %d' % time_slice_count)
        print()

        # create empty files
        get_slice = cached_timeslices(self, time_slice_count)
        # for time_slice in range(time_slice_count):
        #     slice_files[time_slice] = cached_timeslice_init(self, time_slice)

        # compute word frequency
        global_freq = dok_matrix(
            (len(vocab), time_slice_count), dtype=np.uint32)
        mention_freq = dok_matrix(
            (len(vocab), time_slice_count), dtype=np.uint32)

        def print_timing(my_index, start_time):
            now = datetime.now()
            percent_done = my_index / self.size
            elapsed = float((now - start_time).seconds)
            eta_in_seconds = (elapsed / percent_done) * (1 - percent_done)
            eta = now + timedelta(seconds=eta_in_seconds)
            print(
                f'{percent_done:.0%}'.rjust(5),
                f'{my_index}/{self.size}'.rjust(15),
                f'{elapsed:.0f}s'.center(15),
                f'ETA: {eta.strftime("%T")} in {eta_in_seconds:.0f}s')

        start_time = datetime.now()
        my_index = 0
        interval = self.size // 100

        print('Processing tweets...')
        print()
        print('%done', 'curr line/total', 'elapsed seconds',
              'Estimated end time', sep='|')

        print(
            f'{0:.0%}'.rjust(5),
            f'{0}/{self.size}'.rjust(15),
            f'{0:.0f}s'.center(15),
            f'ETA: hh:mm:ss')

        # profile
        import cProfile
        import pstats
        import io
        pr = cProfile.Profile()
        pr.enable()

        for (date, mention, text, words) in self.tokenized_iterator():
            my_index += 1
            if my_index % interval == 0:
                print_timing(my_index, start_time)

            tweet_date = datetime.strptime(date, DATETIME_FORMAT)
            time_slice = get_time_slice_index(tweet_date)
            tweet_count[time_slice] += 1

            for word in set(words):
                word_id = vocab.get(word)
                if word_id is not None:
                    global_freq[word_id, time_slice] += 1
                    if mention:
                        mention_freq[word_id, time_slice] += 1

            # cached_timeslice_append(self, time_slice, text)
            # slice_files[time_slice].write(text + '\n')
            get_slice(time_slice).write(text + '\n')

        # stop profiling
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        global_freq = global_freq.tocsr()
        mention_freq = mention_freq.tocsr()

        return {
            'time_slice_count': time_slice_count,
            'tweet_count': tweet_count,
            'global_freq': global_freq,
            'mention_freq': mention_freq,
        }

    def discretize(self, time_slice_length):
        self.time_slice_length = time_slice_length
        o = self.compute_discretized_corpus()
        self.time_slice_count = o['time_slice_count']
        self.tweet_count = o['tweet_count']
        self.global_freq = o['global_freq']
        self.mention_freq = o['mention_freq']

    def to_date(self, time_slice):
        a_date = self.start_date + timedelta(
            minutes=time_slice * self.time_slice_length)
        return a_date

    def tokenize(self, text):
        # split the documents into tokens based on whitespaces
        raw_tokens = text.split()
        # trim punctuation and convert to lower case
        return [token.strip(string.punctuation).lower() for token in raw_tokens if len(token) > 1 and 'http' not in token]

    def cooccurring_words(self, event, p):
        main_word = event[2]
        slice_start = event[1][0]
        slice_end = event[1][1]  # inclusive

        def words_of(tweet_text: str, main_word: str = None):
            words = self.tokenize(tweet_text)
            if (main_word is None) or (main_word in words):
                for word in words:
                    if word != main_word and len(word) > 1 and word in self.vocabulary:
                        yield word

        word_frequency = {}
        for i in range(slice_start, slice_end + 1):
            for tweet_text in cached_timeslice_read(self, i):
                for word in words_of(tweet_text, main_word):
                    # increment word frequency
                    if word not in word_frequency:
                        word_frequency[word] = 0
                    word_frequency[word] += 1

        # sort words w.r.t frequency
        vocabulary = list(word_frequency.items())
        vocabulary.sort(key=operator.itemgetter(1), reverse=True)
        top_cooccurring_words = []
        for word, frequency in vocabulary:
            top_cooccurring_words.append(word)
            if len(top_cooccurring_words) == p:
                # return the p words that co-occur the most with the main word
                return top_cooccurring_words

        # TODO
        # # ??? return the cooccurring words even if there are less than p words
        # return top_cooccurring_words

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
        # self.min_date = datetime.strptime(self.min_date, DATETIME_FORMAT)

        # load stop-words
        self.stopwords = utils.load_stopwords(stopwords_file_path)

        vocab_vector = self.compute_vocabulary_vector()
        if save_voc:
            utils.write_vocabulary(vocab_vector)

        self.start_date = datetime.strptime(self.start_date, DATETIME_FORMAT)
        self.end_date = datetime.strptime(self.end_date, DATETIME_FORMAT)

        # construct the vocabulary map
        self.vocabulary = self.compute_filtered_vocabulary_map(
            min_absolute_freq, max_relative_freq, vocab_vector)
        print('   Filtered vocabulary: %d distinct words' %
              len(self.vocabulary))

        print('   Corpus: %i tweets, spanning from %s to %s' %
              (self.size, self.start_date, self.end_date))

    def compute_filtered_vocabulary_map(self, min_absolute_freq, max_relative_freq, vocab_vector):
        vocab_map = {}
        word_index = 0
        for word, frequency in vocab_vector:
            if frequency > min_absolute_freq and float(frequency / self.size) < max_relative_freq and word not in self.stopwords:
                vocab_map[word] = word_index
                word_index += 1
        return vocab_map

    def compute_vocabulary_vector(self):
        # identify features
        word_frequency = {}
        for (date, text) in self.source_csv_iterator():
            self.size += 1

            words = self.tokenize(text)

            if date > self.end_date:
                self.end_date = date
            elif date < self.start_date:
                self.start_date = date

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

        return vocabulary

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

                yield date, text

    def import_discretized(self, file_path, time_slice_length):
        if os.path.exists(file_path):
            print(f'Importing previous corpus from {file_path} and corpus/')
            try:
                data = utils.load_pickle(file_path)
                self.time_slice_length = data['time_slice_length']
                self.time_slice_count = data['time_slice_count']
                self.tweet_count = data['tweet_count']
                self.global_freq = data['global_freq']
                self.mention_freq = data['mention_freq']

                if data['source_file_path'] != self.source_file_path:
                    raise Exception('Warning: .source_file_path mismatch')
                if data['size'] != self.size:
                    raise Exception('Warning: .size mismatch')
                if data['start_date'] != self.start_date:
                    raise Exception('Warning: .start_date mismatch')
                if data['end_date'] != self.end_date:
                    raise Exception('Warning: .end_date mismatch')
                if data['min_date'] != self.min_date:
                    raise Exception('Warning: .min_date mismatch')

                if time_slice_length != self.time_slice_length:
                    print('Warning: stored corpus has different time_slice_length')
                    raise Exception(
                        f'Expected: time_slice_length = {time_slice_length}, Got: {self.time_slice_length}')

                for i in range(self.time_slice_count):
                    if not utils.time_slice_exists(i):
                        raise Exception(f'Missing corpus time slice file: {i}')

                return True

            except Exception as e:
                print('Error while importing discretized corpus')
                print(e)

        print('Doing discretization now')
        self.discretize(time_slice_length)
        self.save_discretized(file_path)
        return False

    def save_discretized(self, file_path):
        data = utils.pick_fields(self, [
            'time_slice_count', 'tweet_count', 'global_freq', 'mention_freq', 'time_slice_length',
            'source_file_path', 'size', 'start_date', 'end_date', 'min_date'])
        utils.save_pickle(data, file_path)

    def discretize(self, time_slice_length):
        import spacy

        self.time_slice_length = time_slice_length

        nlp = spacy.load("en_core_web_sm")

        # clean the data directory
        if os.path.exists('corpus'):
            shutil.rmtree('corpus')
        os.makedirs('corpus')

        # compute the total number of time-slices
        time_delta = (self.end_date - self.start_date)
        time_delta = time_delta.total_seconds()/60
        self.time_slice_count = int(time_delta // self.time_slice_length) + 1
        self.tweet_count = np.zeros(self.time_slice_count)
        print(' Number of time-slices: %d' % self.time_slice_count)
        print()

        # create empty files
        for time_slice in range(self.time_slice_count):
            utils.init_time_slice(time_slice)

        # compute word frequency
        self.global_freq = dok_matrix(
            (len(self.vocabulary), self.time_slice_count), dtype=np.uint32)
        self.mention_freq = dok_matrix(
            (len(self.vocabulary), self.time_slice_count), dtype=np.uint32)

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

        for (date, text) in self.source_csv_iterator():
            my_index += 1
            if my_index % interval == 0:
                print_timing(my_index, start_time)

            # tokenize the tweet and update word frequency
            words = self.tokenize(text)

            # mention = '@' in text
            # mention = 'Apple' in text

            nlp_text = nlp(text)

            # propnouns = filter(lambda t: t.pos_ == 'PROPN', nlp_text)
            # has_propnouns = any(propnouns)

            orgs = filter(lambda t: t.ent_type_ ==
                          'ORG' and len(t.text) > 1, nlp_text)
            # At least 2 organizations mentionned
            has_orgs = any(orgs) and any(orgs)

            mention = has_orgs

            # if mention:
            #     print(text)

            tweet_date = datetime.strptime(date, DATETIME_FORMAT)
            time_delta = (tweet_date - self.start_date)
            time_delta = time_delta.total_seconds() / 60
            time_slice = int(time_delta / self.time_slice_length)
            self.tweet_count[time_slice] += 1

            for word in set(words):
                word_id = self.vocabulary.get(word)
                if word_id is not None:
                    self.global_freq[word_id, time_slice] += 1
                    if mention:
                        self.mention_freq[word_id, time_slice] += 1

            utils.append_to_time_slice(time_slice, text)

        self.global_freq = self.global_freq.tocsr()
        self.mention_freq = self.mention_freq.tocsr()

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
            for tweet_text in utils.read_time_slice(i):
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

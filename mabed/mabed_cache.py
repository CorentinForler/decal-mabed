from ctypes import Union
from operator import itemgetter
import pickle
from enum import Enum
from hashlib import md5
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mabed.corpus import Corpus
    from mabed.mabed import MABED


class CacheLevel(Enum):
    L0_GLOBAL = 0  # file shared by all executions
    L1_DATASET = 1  # file shared for a specific dataset (and separator)
    L2_VOCAB = 2  # file shared for a specific vocabulary, MAF and MRF parameters
    L3_DISCRETE = 3  # file shared for a specific corpus and vocabulary, MAF and MRF parameters
    L4_MABED = 4  # file shared for a specific mabed run


class Hash:
    @staticmethod
    def file(path: str):
        hash_md5 = md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def h(*args):
        return md5(':'.join(map(str, args)).encode('utf-8')).hexdigest()

    @staticmethod
    def all(*args, **kwargs):
        import re

        def clean(s: str):
            return re.sub(r'[^\w.-]', '_', s)

        segments = []
        for a in args:
            segments.append(clean(str(a)))

        for k, v in sorted(kwargs.items(), key=itemgetter(0)):
            segments.append(clean(k) + '=' + clean(str(v)))

        return ','.join(segments)


BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')


def cached_getpath(c: 'Corpus', level: CacheLevel, filename: str, ext: str = '.pkl', mabed: 'MABED' = None):
    dataset_hash = Hash.file(c.source_file_path)
    vocab_hash = Hash.all(maf=c.min_absolute_freq,
                          mrf=c.max_relative_freq)
    corpus_hash = Hash.all(tsl=c.time_slice_length)
    mabed_hash = Hash.all(
        k=mabed.k,
        p=mabed.p,
        s=mabed.sigma,
        t=mabed.theta
    ) if mabed else 'NO_MABED'

    if level == CacheLevel.L0_GLOBAL:
        return f"{BASE_PATH}/{filename}{ext}"
    elif level == CacheLevel.L1_DATASET:
        return f"{BASE_PATH}/{dataset_hash}/{filename}{ext}"
    elif level == CacheLevel.L2_VOCAB:
        return f"{BASE_PATH}/{dataset_hash}/{vocab_hash}/{filename}{ext}"
    elif level == CacheLevel.L3_DISCRETE:
        return f"{BASE_PATH}/{dataset_hash}/{vocab_hash}/{corpus_hash}/{filename}{ext}"
    elif level == CacheLevel.L4_MABED and mabed is not None:
        return f"{BASE_PATH}/{dataset_hash}/{vocab_hash}/{corpus_hash}/{mabed_hash}/{filename}{ext}"
    else:
        raise ValueError('Unknown cache level')


def corpus_cached(level: CacheLevel, filename: str):
    filename = filename.replace('/', '_')

    def wrapper(func):
        def wrapped(c: 'Corpus', *args, **kwargs):
            file_path = cached_getpath(c, level, filename)

            if file_path is not None and os.path.isfile(file_path):
                with open(file_path, "rb") as file:
                    print(f"\x1b[1;34mcached\x1b[m {level.name} {filename}")
                    return pickle.load(file)

            result = func(c, *args, **kwargs)

            if file_path is not None:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as file:
                    pickle.dump(result, file)

            return result
        return wrapped
    return wrapper


def mabed_cached(level: CacheLevel, filename: str):
    filename = filename.replace('/', '_')

    def wrapper(func):
        def wrapped(mabed: 'MABED', *args, **kwargs):
            file_path = cached_getpath(
                mabed.corpus, level, filename, mabed=mabed)

            if file_path is not None and os.path.isfile(file_path):
                with open(file_path, "rb") as file:
                    print(f"\x1b[1;34mcached\x1b[m {level.name} {filename}")
                    return pickle.load(file)

            result = func(mabed, *args, **kwargs)

            if file_path is not None:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as file:
                    pickle.dump(result, file)

            return result
        return wrapped
    return wrapper


def cached_timeslice_path(corpus: 'Corpus', index: int):
    dataset_hash = Hash.file(corpus.source_file_path)
    vocab_hash = Hash.all(maf=corpus.min_absolute_freq,
                          mrf=corpus.max_relative_freq)
    corpus_hash = Hash.all(tsl=corpus.time_slice_length)
    return f"{BASE_PATH}/{dataset_hash}/{vocab_hash}/{corpus_hash}/slices/{index}.txt"


def cached_timeslice_init(corpus: 'Corpus', index: int):
    file_path = cached_timeslice_path(corpus, index)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # clear the file
    with open(file_path, 'w', encoding='utf8') as dummy_file:
        dummy_file.write('')

    file = open(file_path, 'w', encoding='utf8')
    return file


def cached_timeslice_append(corpus: 'Corpus', index: int, text: str):
    file_path = cached_timeslice_path(corpus, index)

    with open(file_path, 'a', encoding='utf8') as file:
        file.write(text)


def cached_timeslice_read(corpus: 'Corpus', index: int):
    file_path = cached_timeslice_path(corpus, index)
    with open(file_path, 'r', encoding='utf8') as time_slice_file:
        for tweet_text in time_slice_file:
            yield tweet_text.strip('\n')


class OpenFilesCircularBuffer():
    def __init__(self, max_size: Union[str, int], base_path: str):
        self.max_size = max_size
        self.base_path = base_path
        self.files = {}
        self.keys = []
        self.count = 0

    def close_all(self):
        for file in self.files.values():
            file.close()
        self.files = {}
        self.keys = []
        self.count = 0

    def get(self, key: Union[str, int]):
        if key not in self.files:
            path = f"{self.base_path}/{key}.txt"
            self.files[key] = open(path, 'w', encoding='utf8')
            self.count += 1
            self.keys.append(key)

        if self.count > self.max_size:
            key_to_close = self.keys.pop(0)
            self.files[key_to_close].close()
            del self.files[key_to_close]
            self.count -= 1

        return self.files[key]


def cached_timeslices(corpus: 'Corpus', slices_count: int):
    dir_path = os.path.dirname(cached_timeslice_path(corpus, 0))
    os.makedirs(dir_path, exist_ok=True)

    # create empty files
    for i in range(slices_count):
        file_path = cached_timeslice_path(corpus, i)
        with open(file_path, 'w', encoding='utf8') as dummy_file:
            dummy_file.write('')

    circularbuffer = OpenFilesCircularBuffer(slices_count, dir_path)

    def get(index: int):
        return circularbuffer.get(cached_timeslice_path(corpus, index))

    return get

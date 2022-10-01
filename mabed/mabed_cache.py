from functools import lru_cache
from operator import itemgetter
from enum import Enum
from hashlib import md5
import os
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from mabed.corpus import Corpus
    from mabed.mabed import MABED


import json
import pickle
import marshal
import gc
import time
from datetime import timedelta

PICKLE_EXTENSION = '.pickle'
MARSHAL_EXTENSION = '.marshal'
JSON_EXTENSION = '.json'

DEFAULT_EXTENSION = JSON_EXTENSION


def _cached_dump(data, path):
    gc.disable()

    try:
        start_time = time.monotonic()

        if path.endswith(MARSHAL_EXTENSION):
            with open(path, "wb") as file:
                marshal.dump(data, file)
        elif path.endswith(JSON_EXTENSION):
            with open(path, "w", encoding="utf-8") as file:
                json.dump(data, file,
                          ensure_ascii=False,
                          check_circular=False,
                          allow_nan=False,
                          indent=None, separators=(',', ':'))
        else:
            with open(path, "wb") as file:
                pickle.dump(data, file, protocol=-1)

        print("WROTE", path, timedelta(seconds=time.monotonic() - start_time))
    except Exception as e:
        print("FAILED TO WRITE", path)
        os.remove(path)
        raise
    finally:
        gc.enable()


def _cached_load(file, path):
    if path.endswith(MARSHAL_EXTENSION):
        return marshal.load(file)
    elif path.endswith(JSON_EXTENSION):
        return json.load(file)
    else:
        return pickle.load(file)


@lru_cache(maxsize=None)
def should_not_use_cache():
    # check if global variable named GLOBAL_DISABLE_CACHE exists
    if 'GLOBAL_DISABLE_CACHE' in globals():
        global GLOBAL_DISABLE_CACHE
        return True
    return False


class CacheLevel(Enum):
    L0_GLOBAL = 0  # file shared by all executions
    L1_DATASET = 1  # file shared for a specific dataset
    L2_VOCAB = 2  # file shared for a specific vocabulary, MAF and MRF parameters
    L3_DISCRETE = 3  # file shared for a specific corpus and vocabulary, MAF and MRF parameters
    L4_MABED = 4  # file shared for a specific mabed run

    # comparison operators
    def __eq__(self, other):
        return self.value == other.value

    def __ge__(self, other):
        return self.value >= other.value


class Hash:
    @staticmethod
    @lru_cache(1000)
    def file(path: str):
        hash_md5 = md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    @lru_cache(1000)
    def h(*args):
        return md5(':'.join(map(str, args)).encode('utf-8')).hexdigest()

    @staticmethod
    @lru_cache(1000)
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


def cached_getpath(c: 'Corpus', level: CacheLevel, filename: str, ext: str = DEFAULT_EXTENSION, mabed: 'MABED' = None, extra_data: dict = None):
    path = BASE_PATH

    if level >= CacheLevel.L1_DATASET:
        dataset_hash = Hash.file(c.source_file_path) + '_' + c.min_date_str
        path += f"/{dataset_hash}"

    if level >= CacheLevel.L2_VOCAB:
        vocab_hash = Hash.all(maf=c.min_absolute_freq, mrf=c.max_relative_freq)
        path += f"/{vocab_hash}"

    if level >= CacheLevel.L3_DISCRETE:
        corpus_hash = Hash.all(tsl=c.time_slice_length)
        path += f"/{corpus_hash}"

    if level >= CacheLevel.L4_MABED and mabed is not None:
        mabed_hash = Hash.all(
            k=mabed.k,
            p=mabed.p,
            s=mabed.sigma,
            t=mabed.theta
        )
        path += f"/{mabed_hash}"

    if extra_data is not None:
        extra_data_hash = Hash.all(**extra_data)
        path += f"/{extra_data_hash}"

    if filename and ext:
        path += f"/{filename}{ext}"

    return path


def corpus_cached(level: CacheLevel, filename: str, ext: str = DEFAULT_EXTENSION, extra_data=None):
    if should_not_use_cache():
        return lambda x: x
    filename = filename.replace('/', '_')

    def wrapper(func):
        def wrapped(c: 'Corpus', *args, **kwargs):
            file_path = cached_getpath(
                c, level, filename, ext=ext, extra_data=extra_data)

            if file_path is not None and os.path.isfile(file_path):
                with open(file_path, "rb") as file:
                    print(f"\x1b[1;34mcached\x1b[m {level.name} {filename}")
                    return _cached_load(file, file_path)

            result = func(c, *args, **kwargs)

            if file_path is not None:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                _cached_dump(result, file_path)

            return result
        return wrapped
    return wrapper


def mabed_cached(level: CacheLevel, filename: str, ext: str = DEFAULT_EXTENSION, extra_data=None):
    if should_not_use_cache():
        return lambda x: x
    filename = filename.replace('/', '_')

    def get_file_path(mabed):
        return cached_getpath(mabed.corpus, level, filename, ext=ext, mabed=mabed, extra_data=extra_data)

    def get_cached_result(file_path):
        if file_path is not None and os.path.isfile(file_path):
            with open(file_path, "rb") as file:
                print(f"\x1b[1;34mcached\x1b[m {level.name} {filename}")
                return _cached_load(file, file_path)

    def wrapper(func):
        def wrapped(mabed: 'MABED', *args, **kwargs):
            file_path = get_file_path(mabed)
            result = get_cached_result(file_path)
            if result is not None:
                return result

            result = func(mabed, *args, **kwargs)

            if file_path is not None:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                _cached_dump(result, file_path)
            return result

        def _has_cached_result(*, mabed):
            print(get_file_path(mabed))
            return os.path.isfile(get_file_path(mabed))
        wrapped._has_cached_result = _has_cached_result
        return wrapped
    return wrapper


def cached_timeslice_path(corpus: 'Corpus', index: int):
    return cached_getpath(corpus, CacheLevel.L3_DISCRETE,
                          filename=f'slices/{index}', ext='.txt')


# def cached_timeslice_init(corpus: 'Corpus', index: int):
#     file_path = cached_timeslice_path(corpus, index)
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)

#     # clear the file
#     with open(file_path, 'w', encoding='utf8') as dummy_file:
#         dummy_file.write('')

#     file = open(file_path, 'w', encoding='utf8')
#     return file


def cached_timeslice_read(corpus: 'Corpus', index: int):
    file_path = cached_timeslice_path(corpus, index)
    with open(file_path, 'r', encoding='utf8') as time_slice_file:
        for tweet_text in time_slice_file:
            yield tweet_text.strip('\n')


class TimeSlicesCircularBuffer():
    def __init__(self, max_size: int, base_path: str):
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
        # print(f"\x1b[1;33mcached\x1b[m {key}")
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

    circularbuffer = TimeSlicesCircularBuffer(100, dir_path)

    def get(index: int):
        return circularbuffer.get(index)

    return get, circularbuffer.close_all

# coding: utf-8

# std
import argparse
from mabed.profiler import profile
from textwrap import dedent

# mabed
from mabed.corpus import Corpus
from mabed.mabed import MABED
import mabed.utils as utils

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def parse_arguments():
    p = argparse.ArgumentParser(
        description='Perform mention-anomaly-based event detection (MABED)')

    p.add_argument('i', metavar='input', type=str,
                   help='Input csv file')
    p.add_argument('k', metavar='top_k_events', type=int,
                   help='Number of top events to detect')

    p.add_argument('-o', '--output', metavar='output', type=str,
                   help='Output pickle file', default='output.pickle')
    p.add_argument('-C', '--dont-use-cache', action='store_true',
                   help='Do NOT use a cache (used to improve computation speed on consecutive runs)')

    # Corpus argument group
    c = p.add_argument_group('Corpus discretization')

    # Corpus: Initialization
    c.add_argument('--sw', metavar='stopwords', type=str,
                   help='Stop-word list', default='stopwords/twitter_en.txt')
    # c.add_argument('--sep', metavar='csv_separator', type=str,
    #                help='CSV separator', default='\t')
    c.add_argument('--maf', metavar='min_absolute_frequency', type=int,
                   help='Minimum absolute word frequency, default to 10', default=10)
    c.add_argument('--mrf', metavar='max_relative_frequency', type=float,
                   help='Maximum absolute word frequency, default to 0.4', default=0.4)

    # Corpus: Discretization
    c.add_argument('--tsl', metavar='time_slice_length', type=int,
                   help='Time-slice length, default to 1440 minutes (24 hours)', default=24*60)

    # MABED argument group
    m = p.add_argument_group('Event detection')
    m.add_argument('--p', metavar='p', type=int,
                   help='Number of candidate words per event, default to 5', default=5)
    m.add_argument('--t', metavar='theta', type=float,
                   help='Theta, default to 0.6', default=0.6)
    m.add_argument('--s', metavar='sigma', type=float,
                   help='Sigma, default to 0.6', default=0.6)

    args = p.parse_args()
    return args


def main():
    args = parse_arguments()

    input_path = args.i
    top_k_events = args.k
    stopwords = args.sw
    output_path = args.o
    min_absolute_frequency = args.maf
    max_relative_frequency = args.mrf
    time_slice_length = args.tsl
    p_candidates = args. p
    theta = args.t
    sigma = args.s

    if args.dont_use_cache:
        global GLOBAL_DISABLE_CACHE
        GLOBAL_DISABLE_CACHE = True

    print(dedent(f'''
    Parameters:
        Corpus: {input_path}
        k: {top_k_events}
        Stop-words: {stopwords}
        Min. abs. word freq.: {min_absolute_frequency}
        Max. rel. word freq.: {max_relative_frequency}
        p: {p_candidates}
        theta: {theta}
        sigma: {sigma}
    '''))

    print()
    print('- ' * 10)
    print()

    print('Loading corpus...')
    with utils.timer('Corpus loaded'):
        my_corpus = Corpus(
            input_path,
            stopwords,
            min_absolute_frequency,
            max_relative_frequency)

    print()
    print('- ' * 10)
    print()

    print('Partitioning tweets into %d-minute time-slices...' %
          time_slice_length)
    with utils.timer('Partitioning done'):
        my_corpus.discretize(time_slice_length)

    print()
    print('- ' * 10)
    print()

    print('Running MABED...')
    with utils.timer('Event detection performed'):
        mabed = MABED(my_corpus)
        mabed.run(k=top_k_events, p=p_candidates, theta=theta, sigma=sigma)
        mabed.print_events()

    print()
    if output_path is not None:
        utils.save_events(mabed, output_path)
        print('Events saved in %s' % output_path)


# if __name__ == '__main__':
#     import cProfile
#     import pstats
#     with cProfile.Profile() as pr:
#         main()

#     stats = pstats.Stats(pr)
#     stats.sort_stats(pstats.SortKey.TIME)
#     # Now you have two options, either print the data or save it as a file
#     stats.print_stats()  # Print The Stats
#     # Saves the data in a file, can me used to see the data visually
#     stats.dump_stats("/Users/cogk/Desktop/profile.prof")

if __name__ == '__main__':
    # with profile():
    main()

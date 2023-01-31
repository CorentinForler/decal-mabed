# coding: utf-8

# std
import xlsxwriter
from datetime import datetime
from operator import itemgetter
import argparse

# mabed
from mabed.corpus import Corpus
from mabed.find_articles import find_articles_for_events
from mabed.mabed import MABED
import mabed.utils as utils

__author__ = "Corentin F."
__email__ = ""


def get_mabed(
    *,
    input_path="./stock_article.csv",
    stopwords="./stopwords/custom.txt",
    min_absolute_frequency=10,
    max_relative_frequency=0.2,
    time_slice_length=1440,
    filter_date_after=datetime.strptime("2020-01-01", '%Y-%m-%d'),
):
    my_corpus = Corpus(
        source_file_path=input_path,
        stopwords_file_path=stopwords,
        min_absolute_freq=min_absolute_frequency,
        max_relative_freq=max_relative_frequency,
        filter_date_after=filter_date_after,
    )

    my_corpus.discretize(time_slice_length)

    mabed = MABED(my_corpus)
    return mabed


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Build event browser')
    # p.add_argument('i', metavar='input', type=str,
    #                help='Input pickle file (mabed_phase_2)')
    p.add_argument('-o', '--output', metavar='output',
                   type=str, help='Output csv file')
    args = p.parse_args()

    n_articles = 3

    mabed = get_mabed()
    mabed.run(k=100, p=10, theta=0.6, sigma=0.5)
    # articles = describe_events_okapi_bm25(mabed, mabed.events, n_articles)
    # articles = find_articles_for_events(
    #     mabed, mabed.events, n_articles,
    #     secondary_term_fixed_weight=None,
    #     scoring_method_name='coverage-simple',
    #     divide_score_by_length=False,
    #     use_nlp=False,
    # )  # ~1 minute de calcul
    # articles = find_articles_for_events(
    #     mabed, mabed.events, n_articles,
    #     secondary_term_fixed_weight=None,
    #     scoring_method_name='coverage-count-log',
    #     divide_score_by_length=False,
    #     use_nlp=False,
    # )
    # articles = find_articles_for_events(
    #     mabed, mabed.events, n_articles,
    #     secondary_term_fixed_weight=None,
    #     scoring_method_name='coverage-count-log',
    #     divide_score_by_length=True,
    #     use_nlp=False,
    # )
    articles = find_articles_for_events(
        mabed, mabed.events, n_articles,
        secondary_term_fixed_weight=0.5,
        scoring_method_name='nlp',
        divide_score_by_length=False,
        use_nlp=True,
    )  # 30 minutes de calcul
    # articles = mabed.find_articles_for_events(n_articles=n_articles)
    events = list(utils.iterate_events_as_dict(mabed))

    workbook = xlsxwriter.Workbook(args.output + '.xlsx')
    worksheet = workbook.add_worksheet()

    red = workbook.add_format({
        'color': '#aa0000',
        'bold': True,
        'underline': True
    })
    blue = workbook.add_format({
        'color': '#0055aa',
        'bold': True,
        'italic': True
    })
    text_wrap = workbook.add_format({'text_wrap': True})
    merged_cells = workbook.add_format({
        'text_wrap': True,
        'valign': 'top',
    })

    header = [
        'event id',
        'event main keywords',
        'event secondary keywords',
        'score',
        'proposed rank',
        'text',
        'annotation (is text correct?) : 0, 1',
        'annotation (best one): 0,1'
    ]

    # writer = csv.writer(open(args.output, 'w') if args.output else sys.stdout)
    # writer.writerow(header)
    worksheet.write_row(0, 0, header)

    event_id = 0
    row_index = 0
    for event, articles in zip(events, articles):
        event_id += 1

        main_terms: list = event['term'].split(', ')
        # main_terms.sort()

        related_terms = list(map(itemgetter('term'), event['related']))
        # related_terms.sort()

        worksheet.merge_range(
            f'A{row_index+2}:A{row_index+4}',
            event_id, merged_cells)
        worksheet.merge_range(
            f'B{row_index+2}:B{row_index+4}',
            ', '.join(main_terms), merged_cells)
        worksheet.merge_range(
            f'C{row_index+2}:C{row_index+4}',
            ', '.join(related_terms), merged_cells)

        for (rank, article) in enumerate(articles, start=1):
            row_index += 1

            score, text = article
            row = [
                event_id,
                ', '.join(main_terms),
                ', '.join(related_terms),
                score,
                rank,
                text,
                '',  # annotation
                '',  # annotation
            ]
            # writer.writerow(row)

            worksheet.write_row(row_index, 3, row[3:])

            string_parts = []
            for word in text.split(' '):
                if word.lower() in main_terms:
                    string_parts.append(red)
                elif word.lower() in related_terms:
                    string_parts.append(blue)
                string_parts.append(word + ' ')
            string_parts.append(text_wrap)

            try:
                worksheet.write_rich_string(f'F{row_index+1}', *string_parts)
            except:
                continue

    worksheet.set_column('A:A', 10)
    worksheet.set_column('B:C', 20)
    worksheet.set_column('D:E', 10)
    worksheet.set_column('F:F', 60)

    workbook.close()

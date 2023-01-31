# coding: utf-8

# std
from datetime import datetime
import math
from export_events_to_csv_annotable import get_mabed

# mabed
import mabed.utils as utils

from mabed.sparkline import spark_event

__author__ = "Corentin F."
__email__ = ""


if __name__ == '__main__':
    mabed = get_mabed()
    mabed.run(k=0, p=10, theta=0.6, sigma=0.5)
    events = list(utils.iterate_events_as_dict(mabed))

    for e in events[:3]:
        print(e['start'], e['end'], e['term'])

        print(spark_event(e))
        print()
        print()

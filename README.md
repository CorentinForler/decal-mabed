# pyMABED

## About

pyMABED is a Python 3 implementation of [MABED](#mabed), distributed under the terms of the MIT licence. If you make use of this software in your research, please cite one the [references](#references) below.

## Contributors

Contributors:
- Adrien Guille
- Nicolas Dugué

## MABED

MABED (Mention-Anomaly-Based Event Detection) is a statistical method for automatically detecting significant events that most interest Twitter users from the stream of tweets they publish. In contrast with existing methods, it doesn't only focus on the textual content of tweets but also leverages the frequency of social interactions that occur between users (i.e. mentions). MABED also differs from the literature in that it dynamically estimates the period of time during which each event is discussed rather than assuming a predefined fixed duration for all events.

## Requirements 

pyMABED requires scipy, numpy and networkx; these scientific libraries come pre-installed with the [Anaconda Python](https://anaconda.org) distribution. You can also install them manually via [pip](https://pypi.python.org):  

	pip install scipy
	pip install numpy
	pip install networkx
	
## Usage

Provided a set of tweets, pyMABED can (i) perform event detection and (ii) generate a visualization of the detected events.

### Detect events

detect_events.py has two mandatory (positional) arguments:
 - the path to the csv file containing the tweets,formatted as follows: a tweet per line, with at least a field named 'date' (%Y-%m-%d %H:%M:%S) and a field name 'text' (content of the tweet) ; any other field is ignored
 - the number of top events to detect (e.g. 10)

optional arguments:

  -h, --help            show this help message and exit
  
  --sw stopwords        Stop-word list
  
  --o output            Output pickle file
  
  --maf min_absolute_frequency
                        Minimum absolute word frequency, default to 10
                        
  --mrf max_relative_frequency
                        Maximum absolute word frequency, default to 0.4
                        
  --tsl time_slice_length
                        Time-slice length, default to 30 (minutes)
                        
  --p p                 Number of candidate words per event, default to 10
  
  --t theta             Theta, default to 0.6
  
  --s sigma             Sigma, default to 0.6
  
  --sep separator       Separator, default to `\t` (tab)
  
By default, detect_events.py prints the descriptions of the detected events in the terminal. In order to generate the visualization, you have to run detect_events.py with the --o argument and specify a path.  

Event detection is not multithreaded. Multiple instances on different datasets must be run from own their working directories each, because temporary files are kept in a `corpus` directory inside the cwd. If your data includes emoji or other multibyte characters, be sure to remove them beforehand or use the correct encodings for the csv.

### Visualize events

build_event_browser.py has one mandatory (positional) argument:

- the path to the file that describes the events to visualize (i.e. the path that was passed to detect_events.py with the --o argument) 

optional arguments:

  -h, --help  show this help message and exit
  
  --o output  Output html directory

By default, build_event_browser.py starts a local Web server accessible at http://localhost:2016/. If a path is passed with the --o argument, the visualization is saved on disk in html format.

## MABED

MABED (Mention-Anomaly-Based Event Detection) is a statistical method for automatically detecting significant events that most interest Twitter users from the stream of tweets they publish. In contrast with existing methods, it doesn't only focus on the textual content of tweets but also leverages the frequency of social interactions that occur between users (i.e. mentions). MABED also differs from the literature in that it dynamically estimates the period of time during which each event is discussed rather than assuming a predefined fixed duration for all events.

## References

- Adrien Guille and Cécile Favre (2015) 
  [Event detection, tracking, and visualization in Twitter: a mention-anomaly-based approach](https://github.com/AdrienGuille/pyMABED/blob/master/mabed.pdf).
  Springer Social Network Analysis and Mining,
  vol. 5, iss. 1, art. 18 [DOI: 10.1007/s13278-015-0258-0]


- Adrien Guille and Cécile Favre (2014) 
  Mention-Anomaly-Based Event Detection and Tracking in Twitter.
  In Proceedings of the 2014 IEEE/ACM International Conference on
  Advances in Social Network Mining and Analysis (ASONAM 2014),
  pp. 375-382 [DOI: 10.1109/ASONAM.2014.6921613]
  

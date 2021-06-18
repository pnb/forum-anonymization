# coding=utf-8
# The strategy here is to:
# 1) Subtract a list of English names from a dictionary to get a word list of non-name English
# 2) Further subtract place names (locations) from the word list
# 3) Subtract the word list from the words in forum posts -- whatever remains could be names/places
import sys
import argparse
import re
from collections import OrderedDict, deque

import pandas as pd
import numpy as np


PRE_CONTEXT_LENGTH = 5  # How many words before the current possible name considered as context
MAX_CONTEXT_WORDS = 10  # The top N most common context words will be kept
EMAIL_REGEX = r'\b[a-zA-Z0-9_.+]+@[a-zA-Z0-9_.+]+\.[a-zA-Z0-9_.+]+\b'
# We had a URL regex from https://daringfireball.net/2010/07/improved_regex_for_matching_urls
# But it caused infinite loops so this simple one will have to do
URL_REGEX = r'\b((http|https|ftp)://|www\d{0,3}.)\S+'
# Phone regex adapted from https://stackoverflow.com/questions/16699007
PHONE_REGEX = r'(\b|\()(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
NUMBERS_REGEX = r'\b[0-9]+\b'


assert 1 / 2 > 0, 'This script requires Python 3'

argparser = argparse.ArgumentParser(
    description='Performs step 1 of forum data anonymization: extracting possible name words.')
argparser.add_argument('input_filename', type=str,
                       help='Filename of a CSV file with two columns. The first column should be a '
                       'forum post ID column (not used until the step 2 script) and the second '
                       'column should contain the text of the forum posts. Alternatively, the file '
                       'can be an XLSX file with a "post" ID column and a "message_text" column, '
                       'both in a "post" sheet.')
argparser.add_argument('output_filename', type=str,
                       help='Output will be written in CSV format to this file. The file will be '
                       'overwritten if it already exists.')
args = argparser.parse_args()

def find_number_of_capitalized_letters(word):
    capitalized_letter_count = 0
    for letter in word:
        if letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            capitalized_letter_count += 1
    return capitalized_letter_count

def find_end_of_sentence_occurences(w, post):
    count = 0
    # Splitting post into sentences
    sentences = post.split('.')
    for sentence in sentences:
        # Splitting a sentence into words
        words = sentence.split(' ')
        # Cgecking if the given word occurs at the end of the sentence.
        if w == words[len(words) - 1].lower():
            count += 1
    return count

print('Loading dictionary data')
# English dictionary from Ubuntu 18.04 "wamerican" package; discard names and possessives
wordlist = set()
with open('ubuntu-american-english.txt', encoding='utf-8') as dictfile:
    wordlist.update([w.strip().lower() for w in dictfile if "'s" not in w and
                     (w[0].lower() == w[0] or w.strip()[-1].upper() == w.strip()[-1])])
dictionary = set(wordlist)  # Make a copy to later fill is_dictionary_word
print(str(len(wordlist)) + ' dictionary words loaded')

# Subtract a list of names from a dictionary to get a dictionary of non-name words
# Name list from: http://deron.meranda.us/data/
first_names = pd.read_csv('census-derived-all-first.txt', sep=r'\s+', header=None, na_filter=False)
first_names = {row[0].lower(): row[1] for _, row in first_names.iterrows()}
last_names = pd.read_csv('census-dist-2500-last.txt', sep=r'\s+', header=None, na_filter=False)
last_names = {row[0].lower(): row[1] for _, row in last_names.iterrows()}
wordlist.difference_update(np.concatenate([list(first_names.keys()), list(last_names.keys())]))
print(str(len(wordlist)) + ' non-name words remain after removing common names')

# Subtract location names from the wordlist as well
# Comes from geonames.org, as curated in https://github.com/datasets/world-cities
places = pd.read_csv('world-cities.csv')
cities = places.name.str.lower().astype(str).unique()
countries = places.country.str.lower().astype(str).unique()
subcountries = places.subcountry.str.lower().astype(str).unique()
wordlist.difference_update(np.concatenate([cities, countries, subcountries]))
print(str(len(wordlist)) + ' non-name words remain after removing location names')

print('Loading forum posts')
if args.input_filename.endswith('.csv'):
    with open(args.input_filename, 'r', encoding='utf-8', errors='backslashreplace') as infile:
        posts = pd.read_csv(infile)
else:
    posts = pd.read_excel(args.input_filename, sheet_name='post')
    posts = posts[['post', 'message_text']]
if len(posts.columns) != 2:
    print('Input CSV file must contain two columns')
    sys.exit(1)
post_ids = list(posts[posts.columns[0]].astype(str))
posts = list(posts[posts.columns[1]].astype(str))
print(str(len(posts)) + ' posts found')
# Remove ASCII chars > 127 (for latin1 encoding option)
# posts = [''.join([c for c in p if ord(c) < 128]) for p in posts]

print('Preprocessing multi-word place names')
multi_word_names = [p for p in np.concatenate([cities, countries, subcountries]) if ' ' in p]
print(str(len(multi_word_names)) + ' multi-word place names found')
# Remove spaces from place names in posts
for post_i, post in enumerate(posts):
    if post_i % 100 == 0:
        print('%.1f%%' % (post_i / len(posts) * 100), end='\r')
    post_lowercase = post.lower()  # Avoid doing this multiple times
    for place in multi_word_names:
        if place in post_lowercase:
            post = posts[post_i] = re.sub(place, place.replace(' ', ''), post, flags=re.IGNORECASE)
            post_lowercase = post.lower()
# Now remove spaces in the lists of place names so they still match in the actual posts
cities = [p.replace(' ', '') for p in cities]
countries = [p.replace(' ', '') for p in countries]
subcountries = [p.replace(' ', '') for p in subcountries]
multi_word_map = {p.replace(' ', ''): p for p in multi_word_names}

print('Processing posts')
mentions = {}
for post_i, (post_id, post) in enumerate(zip(post_ids, posts)):
    print('Post ID: %s (%.1f%%)' % (post_id, 100 * post_i / len(posts)), end='\r')
    # Remove email addresses
    post = re.sub(EMAIL_REGEX, ' email_placeholder ', post)
    # Remove URLs
    post = re.sub(URL_REGEX, ' url_placeholder ', post)
    # Remove phone numbers
    post = re.sub(PHONE_REGEX, ' phone_placeholder ', post)
    # Replace other numbers (could be addresses/zip codes, also just not very semantic)
    post = re.sub(NUMBERS_REGEX, ' number_placeholder ', post)
    # Make non-possessive contractions parse as whole words (e.g., should've -> shouldve)
    post = re.sub(r'\b([a-zA-Z]+)\'([a-rt-zA-RT-Z]+)\b', r'\1_\2', post)
    # Tokenize and process individual words
    prev_context = deque(maxlen=PRE_CONTEXT_LENGTH)
    words = re.split(r'\W+', post)
    for word_i, w in enumerate(words):
        w = w.strip()
        uppercase = int(w[0] == w[0].upper()) if len(w) > 0 else 0
        case_preserved_word = w
        w = w.lower()
        if not re.match(r'.*[0-9].*', w) and len(w) > 1 and w not in wordlist and '_' not in w:
            # Could be a name
            if w not in mentions: 
                mentions[w] = OrderedDict({
                    'possible_name': w,
                    'multi_word_name_original': multi_word_map[w] if w in multi_word_map else '',
                    'index_in_post': word_i,
                    'avg_index_in_post': 0, # Avg will be calculated after recording all the occurences
                    'post_length_words': len(words),
                    'avg_post_length_word': 0, # Avg will be calculated after recording all the occurences
                    'capitalized_letters_count': 0, # Avg will be calculated after recording the sum of capital letter in all occurences
                    'occurrences': 0,
                    'capitalized_occurrences': 0,
                    'mid_sentence_cap': 0,
                    'sentence_start_occurrences': 0,
                    'sentence_end_occurences': 0,
                    'is_dictionary_word': int(w in dictionary),
                    'common_firstname_freq': first_names[w] if w in first_names else 0,
                    'common_lastname_freq': last_names[w] if w in last_names else 0,
                    'is_city': int(w in cities),
                    'is_country': int(w in countries),
                    'is_subcountry': int(w in subcountries),
                    'context_words': {},  # Context before all occurrences
                    'context_words_capital': {},  # Context only before capitalized occurrences
                    'context_words_mid_cap': {},  # Context only for mid-sentence capitalizations
                    'occurence_indices':[], # Recording the occurence indices for the word
                    'post_length_counts':[], # Recording the length of posts in which the word occurs
                })
            mentions[w]['occurrences'] += 1
            mentions[w]['capitalized_letters_count'] += find_number_of_capitalized_letters(case_preserved_word)
            mentions[w]['occurence_indices'].append(word_i)
            mentions[w]['post_length_counts'].append(len(words))
            mentions[w]['sentence_end_occurences'] += find_end_of_sentence_occurences(w, post)
            # Sentence start occurrence calculation may be slightly approximate
            sentence_start = len(prev_context) == 0 or \
                re.match('.*' + prev_context[-1] + r'[?.!]\s+' + w + '.*', post, re.IGNORECASE)
            if sentence_start:
                mentions[w]['sentence_start_occurrences'] += 1
            if uppercase:
                mentions[w]['capitalized_occurrences'] += 1
                if not sentence_start:
                    mentions[w]['mid_sentence_cap'] += 1
            for context in prev_context:  # Increment count of BoW context
                if context not in mentions[w]['context_words']:
                    mentions[w]['context_words'][context] = 0
                mentions[w]['context_words'][context] += 1
                if uppercase:
                    if context not in mentions[w]['context_words_capital']:
                        mentions[w]['context_words_capital'][context] = 0
                    mentions[w]['context_words_capital'][context] += 1
                    if not sentence_start:
                        if context not in mentions[w]['context_words_mid_cap']:
                            mentions[w]['context_words_mid_cap'][context] = 0
                        mentions[w]['context_words_mid_cap'][context] += 1
        prev_context.append(w)

print('Finding most frequent context words')
for key in ['context_words', 'context_words_capital', 'context_words_mid_cap']:
    for w in mentions:
        # Computes the mean occurence index of the word in all the posts
        avg_word_index = sum(mentions[w]['occurence_indices']) / mentions[w]['occurrences']
        mentions[w]['avg_index_in_post'] = avg_word_index

        # Computes the mean length of the posts in which the word occurs
        avg_post_length = sum(mentions[w]['post_length_counts']) / mentions[w]['occurrences']
        mentions[w]['avg_post_length_word'] = avg_post_length

        if mentions[w][key]:
            mentions[w][key]['possible_name'] = \
                sum(mentions[w][key][v] for v in mentions[w][key] if v in mentions)
        top_words = sorted(mentions[w][key].items(), key=lambda v: v[1], reverse=True)
        top_words = [v[0] for v in top_words if v[0] not in mentions][:MAX_CONTEXT_WORDS]
        mentions[w][key] = ' '.join(top_words)

print('Saving results')
df = pd.DataFrame.from_records(list(mentions.values()))
df.insert(1, 'num_posts', len(posts))  # Included for ease of later feature calculation
# Dropping the intermediate fields that were used to compute a different feature
df.drop(['occurence_indices'], axis=1, inplace = True)
df.drop(['post_length_counts'], axis=1, inplace = True)
df.to_csv(args.output_filename, index=False)

# Make a dataset suitable for supervised classification, from the list of possible names. This is
# basically the feature extraction step.
from collections import OrderedDict
import re
import argparse

import pandas as pd
import numpy as np


MAX_CONTEXT_WORDS = 25  # Dummy-code occurrences of the most common N context words


argparser = argparse.ArgumentParser(
    description='Compute some features from the list of possible names, and align ground truth if '
                'provided (for training models)')
argparser.add_argument('possible_names_csv', type=str,
                       help='Filename of a CSV file that was created as output from '
                       'anonymize_forums_step1.py')
argparser.add_argument('output_filename', type=str,
                       help='Output will be written in CSV format to this file; the file will be '
                       'overwritten if it already exists')
argparser.add_argument('--ground-truth', type=str, default='', metavar='CSV_FILENAME',
                       help='Filename of a CSV file with a "possible_name" column and an "is_name" '
                       'column that has ground truth (1 for name, 0 for not name)')
argparser.add_argument('--match-context-file', type=str, default=None, metavar='CSV_FILENAME',
                       help='Optional features file previously created using this script, which '
                       'will be used to determine which word occurrence features to include '
                       '(useful for making two compatible datasets)')
args = argparser.parse_args()


print('Loading dictionary data for spell check features')
# English dictionary from Ubuntu 18.04 "wamerican" package; discard names and possessives
wordlist = set()
with open('ubuntu-american-english.txt', encoding='utf-8') as dictfile:
    wordlist.update([w.strip().lower() for w in dictfile if "'s" not in w and
                     (w[0].lower() == w[0] or w.strip()[-1].upper() == w.strip()[-1])])
dictionary = set(wordlist)  # Make a copy to later fill is_dictionary_word
print(str(len(wordlist)) + ' dictionary words loaded')
# Subtract a list of names from a dictionary to get a dictionary of non-name words
# Name list from: http://deron.meranda.us/data/
first_names = \
    pd.read_csv('census-derived-all-first.txt', sep=r'\s+', header=None)[0].str.lower().values
last_names = pd.read_csv('census-dist-2500-last.txt', sep=r'\s+', header=None)[0].str.lower().values
wordlist.difference_update(np.concatenate([first_names, last_names]))
print(str(len(wordlist)) + ' non-name words remain after removing common names')


# Spell check code roughly based on http://norvig.com/spell-correct.html
# Here we allow the alphabet to be any letters that occur in our word list (more than 26)
alphabet = set(l for w in wordlist for l in w) - {"'"}


def known(words):
    # The subset of `words` that appear in the dictionary
    return set(w for w in words if w in wordlist)


def edits1(word):
    # Set of edits that are one edit away from `word`
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in alphabet]
    inserts = [L + c + R for L, R in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    # Set of edits that are two edits away from `word`
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))


print('Loading possible names data')
df = pd.read_csv(args.possible_names_csv, encoding='utf-8', na_filter=False)  # "Nan" != NA
if args.ground_truth:
    gt = pd.read_csv(args.ground_truth, encoding='utf-8', na_filter=False)
    gt = {w: t for w, t in gt[['possible_name', 'is_name']].values}
    df['truth'] = [gt[w] for w in df.possible_name.values]  # Map truth values onto data

common_context = set()
if args.match_context_file is None:  # Infer from word frequencies
    print('Calculating frequency of context words')
    context_freq = {}
    for _, row in df.iterrows():
        cw = set(str(row.context_words).split(' ')) | \
            set(str(row.context_words_capital).split(' ')) | \
            set(str(row.context_words_mid_cap).split(' '))
        cw -= set(['nan', ''])
        for w in cw:
            context_freq[w] = context_freq[w] + 1 if w in context_freq else 1
    for w, _ in sorted(context_freq.items(), key=lambda v: v[1], reverse=True):
        common_context.add(w)
        if len(common_context) == MAX_CONTEXT_WORDS:
            break
else:  # Load from another dataset file so the columns will match
    print('Loading list of common context words from file:', args.match_context_file)
    for col in pd.read_csv(args.match_context_file, encoding='utf-8', na_filter=False):
        if col.startswith('context_'):
            common_context.add(col[8:])

print('Computing features')
result = []
for row_i, row in df.iterrows():
    if row_i % 10 == 0:
        print('%.1f%%' % (row_i / len(df) * 100), end='\r')
    result.append(OrderedDict({
        'possible_name': row.possible_name,
        'multi_word_name_original': row.multi_word_name_original,
        'is_name': row.truth if 'truth' in row.index else '',
        'occurrences': row.occurrences,  # / row.num_posts,  # Prop. probably won't generalize
        'first_index_in_post': row.index_in_post,
        'first_post_length_words': row.post_length_words,
        'prop_capitalized': row.capitalized_occurrences / row.occurrences,
        'prop_sentence_start': row.sentence_start_occurrences / row.occurrences,
        'prop_mid_sentence_cap': row.mid_sentence_cap /
        (row.occurrences - row.sentence_start_occurrences)
        if row.occurrences - row.sentence_start_occurrences > 0 else -1,
        'prop_mid_sentence_cap_overall': row.mid_sentence_cap / row.occurrences,
        'is_dictionary_word': row.is_dictionary_word,
        'is_common_firstname': int(row.common_firstname_freq > 0),
        'is_common_lastname': int(row.common_lastname_freq > 0),
        'common_firstname_freq': row.common_firstname_freq,
        'common_lastname_freq': row.common_lastname_freq,
        'is_city': row.is_city,
        'is_country': row.is_country,
        'is_subcountry': row.is_subcountry,
        'num_words_edit1': len(known(edits1(row.possible_name))),
        'num_words_edit2': len(known(edits2(row.possible_name))),
    }))
    # Dummy-coded context words occurrence, with capitalized words separate
    for cw, cap in [(row.context_words, ''), (row.context_words_capital, 'cap_'),
                    (row.context_words_mid_cap, 'mid_cap_')]:
        cw = set(str(cw).split(' ')) - set(['nan', ''])
        for w in sorted(common_context):
            result[-1][cap + 'context_' + w] = 1 if w in cw else 0
            cw.discard(w)
        result[-1][cap + 'other_context'] = len(cw)  # Count of remaining uncommon context words

print('Saving result')
pd.DataFrame.from_records(result).to_csv(args.output_filename, index=False)
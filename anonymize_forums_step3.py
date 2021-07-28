# Takes in a list of forum posts and a list of probable names, and removes the names from the posts.
# Names are replaced with one-way hashed placeholders.
import sys
import re
from collections import OrderedDict
import argparse
from itertools import repeat

import pandas as pd
from tqdm import tqdm
import multiprocessing

def yes_no_prompt(prompt_text):
    ans = ''
    while ans.strip() not in ['y', 'n']:
        ans = input(prompt_text + ' [y/n] ')
    return ans == 'y'

def redact_post(index, post, regex_dict):
    # Remove email addresses
    p = regex_dict['email'].sub(' email_placeholder ', post)
    # Remove URLs
    p = regex_dict['url'].sub(' url_placeholder ', p)
    # Remove phone numbers
    p = regex_dict['phone'].sub(' phone_placeholder ', p)
    # Replace other numbers (could be addresses/zip codes, also just not very semantic)
    p = regex_dict['number'].sub(' number_placeholder ', p)
    # Remove names
    p = regex_dict['name'].sub('name_placeholder', p)
    return int(index), p
    
def main():

    ids = []

    # See step 1 for info on regular expressions (TODO: ideally these should not be redundant)
    EMAIL_REGEX = re.compile(r'\b[a-zA-Z0-9_.+]+@[a-zA-Z0-9_.+]+\.[a-zA-Z0-9_.+]+\b')
    URL_REGEX = re.compile(r'\b((http|https|ftp)://|www\d{0,3}.)\S+')
    PHONE_REGEX = re.compile(r'(\b|\()(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b')
    NUMBERS_REGEX = re.compile(r'\b[0-9]+\b')

    # Check known requirements
    assert 1 / 2 > 0, 'This script requires Python 3'
    assert pd.__version__ >= '0.22.0', 'This script requires Pandas version 0.22.0 or higher'

    # Check CLI usage
    argparser = argparse.ArgumentParser(
        description='Performs redaction of identified names and other identifying info in forum text')
    argparser.add_argument('names_file', type=str,
                        help='Filename of a CSV file with a "possible_name" columns specifying names'
                        ' to remove (output of step 2b)')
    argparser.add_argument('posts_file', type=str,
                        help='Filename of a CSV file with two columns, the first for post IDs and '
                        'the second with the actual forum posts; alternatively, this may be an XLSX '
                        'file with a "post" ID column and a "message_text" column, both in a "post" '
                        'sheet (input to step 1)')
    argparser.add_argument('output_file', type=str,
                        help='Output (anonymized forum posts) will be written as CSV to this file.')
    args = argparser.parse_args()

    # Read in data and do some sanity checks
    print('Loading data')
    if args.posts_file.endswith('.csv'):
        with open(args.posts_file, 'r', encoding='utf-8', errors='backslashreplace') as infile:
            df = pd.read_csv(infile, na_filter=False)
    else:
        df = pd.read_excel(args.posts_file, sheet_name='post')
        df = df[['post', 'message_text']]
    assert len(df.columns) == 2, 'Input CSV file must contain two columns'
    ids = df[df.columns[0]].astype(str)
    posts = df[df.columns[1]].astype(str)

    if sum(len(x) for x in ids) > sum(len(x) for x in posts):
        print('The average length of values in the ID column (first column) is longer than the'
            '\naverage length of values in the discussion forum post column.')
        yes_no_prompt('Are you sure this is correct?') or sys.exit(1)

    if len(ids) != len(ids.unique()):
        print('The IDs specified in the ID column (first column) are not unique.')
        yes_no_prompt('Are you sure the input file is correct?') or sys.exit(1)

    names_df = pd.read_csv(args.names_file, encoding='utf-8', na_filter=False)
    if 'possible_name' not in names_df.columns:
        print('The file with a list of possible names must contain a "possible_name" column.')
        sys.exit(1)
    names = names_df.possible_name.values
    names_regex = re.compile(r'\b(' + '|'.join(names) + r')\b', flags=re.IGNORECASE)
    
    regex_dict = {
        'email': EMAIL_REGEX,
        'url': URL_REGEX,
        'phone': PHONE_REGEX,
        'number': NUMBERS_REGEX,
        'name': names_regex,
    }
    arguments = list(zip(ids, posts, repeat(regex_dict)))

    with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 2)) as pool:
        # result = pool.starmap(redact_post, arguments)
        result = list(tqdm(pool.starmap(redact_post, arguments), total=len(posts))) 

    # Save results to file
    print('Saving result')
    out_df = pd.DataFrame(result, columns=['id', 'anonymized_post']) 
    out_df.sort_values(by='id',inplace=True)
    out_df.to_csv(args.output_file, index=False, encoding='utf-8')
    
if __name__ == '__main__':
    main()
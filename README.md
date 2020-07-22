# Discussion forum text anonymization

The forum anonymization software consists of several sequential steps, outlined below. The only thing needed to get started is some input data in the correct format. See the step 1 help information (`-h` option) for specifics. An example, `test_posts.csv` is included here.

The software includes pre-trained machine learning models to distinguish names from non-name words, but custom models can be trained between steps 2a and 2b. See the annotation guide, `anonymization_coding_scheme.txt` and the model training code `train_model.py` if custom models are needed.

A detailed description of the anonymization method is available in:

> Bosch, N., Crues, R. W., Shaik, N., & Paquette, L. (2020). "Hello, [REDACTED]": [Protecting student privacy in analyses of online discussion forums](https://educationaldatamining.org/files/conferences/EDM2020/papers/paper_26.pdf). In Proceedings of the 13th International Conference on Educational Data Mining (EDM 2020) (pp. 39–49). International Educational Data Mining Society.

## Set up the python environment

Specific versions of Python packages should be installed to ensure the machine learning models will work. Packages like _Scikit-learn_ and _TensorFlow_ update frequently, and newer versions of these packages are likely incompatible with the machine learning models.

Assuming you have Anaconda or Miniconda installed, run these commands:

    conda create -n forum-redact python==3.7
    conda activate forum-redact
    pip install tensorflow==1.13.1 scikit-learn==0.22 pandas==0.25.3 numpy==1.17.3 scipy==1.3.0 Keras==2.2.4 tqdm==4.40.0

For future use, after installation, the environment can be activated by running `conda activate forum-redact`.

## Step 1: Extract possible names

This step identifies possible names, consisting of words that are either not in the dictionary or are known names. See the help page for details.

    python anonymize_forums_step1.py -h
    python anonymize_forums_step1.py test_posts.csv test_posts-step1.csv

## Step 2a: Extract features

This step extracts features that characterize the possible names identified in step 1. If using the pre-trained machine learning models, it is necessary to use the `--match-context-file` option to ensure that the context words match those the model expects, instead of whichever words are most frequent in your specific dataset. If training custom models or evaluating accuracy versus some annotated data on a new dataset, this step can also align annotations (see the `-h` output).

    python anonymize_forums_step2a.py -h
    python anonymize_forums_step2a.py test_posts-step1.csv test_posts-step2a.csv --match-context-file context_template.csv

## Step 2b: Filter down possible names

This step applies the pre-trained machine learning models to identify what is actually a name or not from among the set of possible names identified in step 1.

    python anonymize_forums_step2b.py -h
    python anonymize_forums_step2b.py test_posts-step2a.csv test_posts-step2b.csv

Edit `test_posts-step2b.csv` to remove false positives if needed. The file will be sorted in descending order of word frequency, so egregious false positives will appear toward the top. In the example, _url_ and _numbers_ could probably be removed, though in a typical application only the top few rows with high-frequency, high-impact false positives need to be examined.

## Step 3: Redact names

This step removes the names identified from step 2b from the forum posts, along with other potential identifying information like URLs and numbers.

    python anonymize_forums_step3.py -h
    python anonymize_forums_step3.py test_posts-step2b.csv test_posts.csv test_posts-step3.csv

The output file should now contain the anonymized forum posts.

# Train a model to decide whether a word is a name or not based on context and other features,
# according to the labels provided by humans.
from pprint import pprint
import subprocess
import argparse
import skopt

import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, pipeline, preprocessing
from sklearn import ensemble, tree, naive_bayes, linear_model
from skopt import BayesSearchCV
import joblib

import train_nn_def

RANDOM_SEED = 11798
CACHE_DIR = 'temp'

argparser = argparse.ArgumentParser(description='Train ML models for filtering possible names')
argparser.add_argument('features_file', help='Filename of a features file (output of step 2a)')
argparser.add_argument('output_results_file', help='Cross-validated predictions output CSV name')
argparser.add_argument('output_model_file', help='Pickled final model filename (.pkl)')
argparser.add_argument('model_type', choices=['nn', 'extra-trees'],
                       help='Custom neural network (nn) or Extra-Trees')
args = argparser.parse_args()

print('Loading data')
df = pd.read_csv(args.features_file, encoding='utf-8', na_filter=False)
features = [f for f in df if f not in ['possible_name', 'is_name', 'multi_word_name_original']]
# features = [f for f in features if 'context' not in f]  # Use only non-word (computed) features
# features = [f for f in features if 'context' in f]  # Use only word (context) features
print('Found ' + str(len(features)) + ' features, ' + str(len(df)) + ' instances')
print(features)
assert len(df.is_name.unique()) == 2, df.is_name.unique()
y = df.is_name.values
X = df[features].values
print('Base rate:', np.mean(y))

score_metric = metrics.make_scorer(metrics.mean_squared_error, greater_is_better=False,
                                   needs_proba=True)

if args.model_type == 'extra-trees':
    m = ensemble.ExtraTreesClassifier(n_estimators=500, bootstrap=True, random_state=RANDOM_SEED)
    param_grid = {
        'model__max_features': skopt.space.Real(0.25, 1, prior='uniform'),
        'model__max_samples': skopt.space.Real(0.5, 0.999, prior='uniform'),
        'model__min_samples_leaf': skopt.space.Integer(2, 32, prior='log-uniform', base=2),

        # 'model__max_features': [.25, .5, .75, 1.0],
        # 'model__max_samples': [.5, .75, .999],
        # 'model__min_samples_leaf': [1, 2, 4, 8, 16, 32],
    }
else:
    num_dense_features = features.index([f for f in features if 'context_' in f][0])
    m = train_nn_def.BespokeNN(num_dense_features=num_dense_features, verbose=0,
                               file_prefix=CACHE_DIR + '/holdout_bespoke_nn_')
    param_grid = {
        # 'model__num_hidden_layers': [0, 1, 2, 4],
        # 'model__hidden_layer_size': [2, 4, 8, 16, 32],
        # 'model__dropout': [0, .25, .5],
        # 'model__learning_rate': [.01, .001, .0001],
        # 'model__dense_reg_strength': [.1, .01, .001],  # Bespoke DNN specific parameters
        # 'model__sparse_reg_strength': [.1, .01, .001],

        'model__num_hidden_layers': skopt.space.Integer(0, 4, prior='uniform'),
        'model__hidden_layer_size': skopt.space.Integer(2, 32, prior='log-uniform', base=2),
        'model__dropout': skopt.space.Real(0, 0.5, prior='uniform'),
        'model__learning_rate': skopt.space.Real(.0001, 0.01, prior='log-uniform'),
        'model__dense_reg_strength': skopt.space.Real(.001, 0.1, prior='log-uniform'),  # Bespoke DNN specific parameters
        'model__sparse_reg_strength':skopt.space.Real(.001, 0.1, prior='log-uniform'),
    }

xval = model_selection.StratifiedKFold(4, shuffle=True, random_state=RANDOM_SEED)

pipe = pipeline.Pipeline([
    ('normalize', preprocessing.MinMaxScaler()),  # NN classifier (doesn't hurt trees)
    ('model', m),
])
gs = BayesSearchCV(pipe, param_grid,
                                  n_iter=10,
                                  random_state=RANDOM_SEED,
                                  optimizer_kwargs={'n_initial_points': 5},
                                  scoring=score_metric,
                                  verbose=2,
                                  cv=xval,
                                  refit=True)

threshold = 1 / 3
res = model_selection.cross_validate(gs, X, y, cv=xval, scoring=['roc_auc', 'precision', 'recall'],
                                     verbose=1, return_estimator=True)                                  
pprint({k: v for k, v in res.items() if k != 'estimator'})
print(res)

# Error analysis
err_df = pd.DataFrame(index=df.index, data={'pred': '', 'truth': y, 'fold': ''})
for fold_i, (_, test_i) in enumerate(xval.split(X, y)):
    print('Evaluating fold', fold_i)
    err_df.loc[test_i, 'pred'] = res['estimator'][fold_i].predict_proba(X[test_i]).T[1]
    err_df.loc[test_i, 'fold'] = fold_i
    if isinstance(m, tree.DecisionTreeClassifier):
        # Graph trees, look for the most impure large leaves -- in the tree graphs or in a pivot
        # table filtered by truth value looking for common wrong predicted probabilities
        t = res['estimator'][fold_i].best_estimator_.named_steps['model']
        leaf_i = t.apply(X[test_i])
        leaf_sizes = np.bincount(leaf_i)  # Array with leaf index -> number of occurrences
        if 'leaf_size' not in err_df:
            err_df['leaf_size'] = ''
        err_df.loc[test_i, 'leaf_size'] = [leaf_sizes[i] for i in leaf_i]
        tree.export_graphviz(t, out_file='train_error_analysis_tree.dot',
                             class_names=['not name', 'name'], feature_names=features, filled=True)
        subprocess.call(['dot', '-Tpng', 'train_error_analysis_tree.dot', '-o',
                         'train_error_analysis_tree' + str(fold_i) + '.png', '-Gdpi=300'])
print('AUC =', metrics.roc_auc_score(err_df.truth.values, err_df.pred.values))  # Verify alignment
print('Kappa =', metrics.cohen_kappa_score(err_df.truth, err_df.pred > threshold))
print(len(err_df[(err_df.pred >= threshold) & (err_df.truth == 0)]), 'false positives')
print(len(err_df[(err_df.pred >= threshold) & (err_df.truth == 1)]), 'true positives')
print(len(err_df[(err_df.pred < threshold) & (err_df.truth == 0)]), 'true negatives')
print(len(err_df[(err_df.pred < threshold) & (err_df.truth == 1)]), 'false negatives')
err_df.insert(0, 'possible_name', df.possible_name)
err_df.to_csv(args.output_results_file, index=False)

print('Training on all data')
gs.fit(X, y)
print(gs.best_estimator_)
pprint(gs.best_estimator_)
joblib.dump(gs.best_estimator_, args.output_model_file)

#############################
# Saving the nn pred to compare to the loaded model. This can be removed once testing is complete

# df = pd.read_csv('s2a.csv', encoding='utf-8', na_filter=False)
# features = [f for f in df if f not in ['possible_name', 'is_name', 'multi_word_name_original']]
# new_X = df[features].values

# df.insert(1, 'nn_pred', gs.best_estimator_.predict_proba(new_X).T[1])
# df.to_csv('real_model.csv')
# #############################

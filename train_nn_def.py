# Define neural net model that can be run through sklearn
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, callbacks, regularizers
from tensorflow.keras import layers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.utils.estimator_checks import check_estimator
import numpy as np
import types
import tempfile
import copy
import pandas as pd

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

class NNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_hidden_layers=1, hidden_layer_size=10, dropout=0, learning_rate=.01,
                 loss='binary_crossentropy', epochs=100, validation_prop=.2, batch_size=32,
                 verbose=1, file_prefix=None):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.loss = loss
        self.epochs = epochs
        self.validation_prop = validation_prop
        self.batch_size = batch_size
        self.verbose = verbose
        self.file_prefix = file_prefix if file_prefix is not None else 'tmp-nn-model-'    

    def load_trained_weights(self, model_filename):
        self.model_ = load_model(model_filename)

    def build_model(self, input_shape, output_shape):
        # Override this function in subclassees for more specialized network structures
        m_input = layers.Input(input_shape)
        m = m_input
        if len(input_shape) > 1:
            m = layers.Flatten()(m)  # Simple FC network, flatten everything
        for layer_num in range(self.num_hidden_layers):
            m = layers.Dense(self.hidden_layer_size, name='hidden' + str(layer_num + 1))(m)
            m = layers.LeakyReLU(alpha=.1)(m)
            if self.dropout > 0:
                m = layers.Dropout(self.dropout)(m)
        m_output = layers.Dense(output_shape, activation='sigmoid')(m)
        model = Model(inputs=[m_input], outputs=[m_output])
        opt = optimizers.SGD(self.learning_rate)
        model.compile(opt, loss=self.loss, metrics=['acc'])
        return model

    def fit(self, X, y):
        X, y = check_X_y(X, y)  # Check that X and y have correct shape
        assert np.max(np.abs(X)) <= 3, 'Input values should be in [-3, 3] to avoid certain issues'
        self.classes_ = unique_labels(y)  # Store the classes seen during fit
        self.X_ = X
        self.y_ = y
        cat_y = y
        if len(cat_y.shape) == 1:
            cat_y = to_categorical(cat_y)  # One-hot encode if needed

        K.clear_session()
        self.model_ = self.build_model(X.shape[1:], len(cat_y.shape))
        if self.verbose:
            self.model_.summary()
        model_fname = self.file_prefix + \
            '_'.join([str(v) for k, v in sorted(self.get_params().items()) if k != 'file_prefix'])
        model_fname += '.h5'
        self.model_.fit(X, cat_y,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        validation_split=self.validation_prop,
                        verbose=self.verbose,
                        callbacks=[
                            callbacks.ModelCheckpoint(model_fname, save_best_only=True),
                        ])
        self.model_ = load_model(model_fname)  # Load best val epoch
        return self  # Return the classifier

    def predict_proba(self, X):
        check_is_fitted(self, ['X_', 'y_'])  # Check is fit had been called
        X = check_array(X)  # Input validation
        return self.model_.predict(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class BespokeNN(NNClassifier):
    def __init__(self, num_dense_features, dense_reg_strength=.01, sparse_reg_strength=.01,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_dense_features = num_dense_features
        self.dense_reg_strength = dense_reg_strength
        self.sparse_reg_strength = sparse_reg_strength

    def __getstate__(self):

        # Creating a copy of the dictionary and removing the tensorflow model
        dict =  copy.deepcopy(self.__dict__)
        model = dict.pop('model_')

        # Saving the tensorflow model separately
        model.save_weights('model_weights.h5', overwrite=True)
        open('model_architecture.json', 'w').write(model.to_json())
        
        # Returning the dictionary without the model_ for pickling
        dict['model_'] = ""
        return dict
    
    def __setstate__(self, d):
        
        # Assigning the saved dictionary values after unpickling except the model_
        self.__dict__ = d

        # Rebuilding the model
        m = keras.models.model_from_json(open('model_architecture.json').read())
        m.load_weights('model_weights.h5')
        opt = optimizers.SGD(self.learning_rate)
        m.compile(opt, loss=self.loss, metrics=['acc'])

        # Reassigning the model after building
        self.__dict__['model_'] = m
        

    def get_params(self, deep=False):
        return {
            'num_hidden_layers': self.num_hidden_layers,
            'hidden_layer_size': self.hidden_layer_size,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'loss': self.loss,
            'epochs': self.epochs,
            'validation_prop': self.validation_prop,
            'batch_size': self.batch_size,
            'verbose': self.verbose,
            'file_prefix': self.file_prefix,
            'num_dense_features': self.num_dense_features,
            'dense_reg_strength': self.dense_reg_strength,
            'sparse_reg_strength': self.sparse_reg_strength,
        }

    def build_model(self, input_shape, output_shape):
        m_input = layers.Input(input_shape)
        tmp_ndf = self.num_dense_features  # Prevent Lambda from referencing self
        dense_eng_in = layers.Lambda(lambda x: x[:, :tmp_ndf], name='eng_in')(m_input)
        sparse_word_in = layers.Lambda(lambda x: x[:, tmp_ndf:], name='word_in')(m_input)

        # Engineered features (mostly densely-distributed)
        l2 = regularizers.l2(self.dense_reg_strength)
        m_dense = layers.Dense(self.hidden_layer_size, kernel_regularizer=l2)(dense_eng_in)
        m_dense = layers.LeakyReLU(alpha=.1)(m_dense)
        m_dense = layers.BatchNormalization()(m_dense)

        # Word features (mostly sparse)
        l1 = regularizers.l1(self.sparse_reg_strength)
        m_sparse = layers.Dense(self.hidden_layer_size, kernel_regularizer=l1)(sparse_word_in)
        m_sparse = layers.LeakyReLU(alpha=.1)(m_sparse)
        m_sparse = layers.BatchNormalization()(m_sparse)

        m = layers.Concatenate()([m_dense, m_sparse])
        for layer_num in range(self.num_hidden_layers):
            m = layers.Dense(self.hidden_layer_size, name='hidden' + str(layer_num + 1))(m)
            m = layers.LeakyReLU(alpha=.1)(m)
            if self.dropout > 0:
                m = layers.Dropout(self.dropout)(m)
        m_output = layers.Dense(output_shape, activation='sigmoid', name='output')(m)

        model = Model(inputs=[m_input], outputs=[m_output])
        opt = optimizers.SGD(self.learning_rate)
        model.compile(opt, loss=self.loss, metrics=['acc'])
        return model


if __name__ == '__main__':
    # check_estimator(NNClassifier)  # Fails because of multiclass or something like that

    to_test = 'branch'  # fc or branch

    tmpy = np.random.randint(0, 2, [2000])
    tmpx = np.random.random_sample([2000, 4])
    tmpx[tmpy == 1, 0] += .5  # Add a bit of signal to the noise
    if to_test == 'fc':
        testmodel = NNClassifier(num_hidden_layers=1, hidden_layer_size=50, epochs=100, dropout=.5)
    elif to_test == 'branch':
        testmodel = BespokeNN(num_dense_features=2, dense_reg_strength=.001,
                              sparse_reg_strength=.001, num_hidden_layers=2, hidden_layer_size=8,
                              dropout=.5)
    testmodel.fit(tmpx[:1000], tmpy[:1000])
    print('Train accuracy')
    preds = testmodel.predict(tmpx[:1000])
    print(classification_report(tmpy[:1000], preds))
    print('Kappa =', cohen_kappa_score(tmpy[:1000], preds))
    print('Test accuracy')
    preds = testmodel.predict(tmpx[1000:])
    print(classification_report(tmpy[1000:], preds))
    print('Kappa =', cohen_kappa_score(tmpy[1000:], preds))

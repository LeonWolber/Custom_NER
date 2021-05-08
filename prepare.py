import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

import nltk
import string

import random
from nltk.corpus import stopwords

def prepare_small(ser, set_=False):
    '''
    param ser : pandas series
    param set_ : bool

    Cean text data. Set set_=True if words should only occur once per datapoint

    return Series
    '''
    stop = stopwords.words()
    tokens = ser.apply(lambda x:nltk.word_tokenize(x.strip()))
    remove_stop = tokens.apply(lambda x:[i.strip() for i in x if (i not in string.punctuation.replace('-','')) & (len(i)>1) & (i not in stop)])
    if set_:
        return remove_stop.apply(lambda x: list(set(x)))

    return remove_stop


def prepare_data(high_level_pred=True, set_=False):
    le = LabelEncoder()

    if high_level_pred:
        x = 0
    else:
        x = 1

    # read_data
    df = pd.read_csv('Jira 2021-04-28T17 44 57+0200.csv', sep=';').fillna('')
    df.rename(columns={'Custom field (Department)': 'department', 'Custom field (Function)': 'function',
                       'Custom field (Target Group)': 'target'}, inplace=True)
    # remove groups that occur <= 25 times 
    to_change = df.target.value_counts()[-12:].reset_index()['index']
    df['target'] = pd.Series([i.split('->')[0].strip() if i in list(to_change) else i for i in df.target])

    # concatenate relevant text into one col
    df['full_text'] = df.function + ' ' + df.department

    # create tokenized columns
    df['clean_token_full'] = prepare_small(df['full_text'], set_=set_)
    df['clean_token_func'] = prepare_small(df['function'], set_=set_)

    df['clean_text'] = df['clean_token_full'].apply(lambda x: ' '.join([item for item in x]))
    # frame to predict later
    df_unknown = df[df['target'] == 'Others/None']

    # frame to train on
    df = df[(df.astype(str)['clean_text'] != '') & (df['target'] != 'Others/None')].reset_index()

    # create label column encoded to ints
    df['label'] = le.fit_transform([i.split('->')[x].strip() if '>' in i else i.strip() for i in df['target']])

    return df, df_unknown, le


def create_vecs(df, train=True, vect=None):
    # create sparse matrix representing occurrences of words per document
    # later more sophisticated methods like tf-idf weighting can be applied, however the documents are very short.
    if vect:
        vectorizer = vect
    else:
        vectorizer = CountVectorizer()

    # train test split
    X = df.clean_text.values

    if train:
        y = df.label.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # fit vectorizer on train data
        vectorizer.fit(X_train)
        assert 'Others/None' not in le.classes_

        # create sparse matrices for train,test sentences
        X_train_vect = vectorizer.transform(X_train)
        X_test_vect = vectorizer.transform(X_test)

        return X_train_vect, X_test_vect, y_train, y_test, vectorizer, X_test
    else:
        return vectorizer.transform(X), X


def fit_model(X_train, y_train, X_test, y_test, use_weights=False):
    if use_weights:
        weight = compute_class_weight('balanced', np.unique(y_train), y_train)
        weight_dict = dict(zip(np.unique(y_train), weight))
    else:
        weight_dict = None

    classifier = LogisticRegression(class_weight=weight_dict)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print("Accuracy:", score)
    return classifier



## Create data for Custom NER Model

def create_train_data_multiple(df, enc):
    df = df[df['clean_token_func'].str.len() != 0]
    data = []

    for i in df.iterrows():

        k = random.randint(1, len(i[1].clean_token_func))
        # get sampled word from function
        samples = random.sample(i[1].clean_token_func, k=k)

        # derive index of word in full_text
        start_inds = np.array([i[1].clean_text.index(word) for word in samples])

        inds_len = np.array([len(j) for j in samples])

        end_inds = np.add(start_inds, inds_len)
        #         print(i[1].clean_text)
        assert np.isin(start_inds, end_inds).any() == False

        labels = [enc.inverse_transform(np.array(i[1].label).reshape(-1, 1))[0]] * len(start_inds)
        dict_entry = list(set(zip(start_inds, end_inds, labels)))

        data.append((i[1].clean_text, {'entities': dict_entry}))

    return data


def create_train_data_sequential(df, enc):
    df = df[df['clean_token_func'].str.len() != 0]

    data = []

    for i in df.iterrows():
        k = random.randint(1, len(i[1].clean_token_func))
        # get sampled word from function
        sample = ' '.join(i[1].clean_token_func[:k])

        # derive index of word in full_text
        start_ind = i[1].clean_text.index(sample)

        ind_len = len(sample)

        end_ind = start_ind + ind_len

        label = [enc.inverse_transform(np.array(i[1].label).reshape(-1, 1))[0]]

        data.append((i[1].clean_text, {'entities': [(start_ind, end_ind, label)]}))

    return data


def create_train_data_single(df, enc):
    df = df[df['clean_token_func'].str.len() != 0]

    data = []

    for i in df.iterrows():
        # k = random.randint(1, len(i[1].clean_token_func))
        # get sampled word from function
        sample = random.choice(i[1].clean_token_func)

        # derive index of word in full_text
        start_ind = i[1].clean_text.index(sample[0])

        ind_len = len(sample)

        end_ind = start_ind + ind_len

        label = enc.inverse_transform(np.array(i[1].label).reshape(-1, 1))[0]

        data.append((i[1].clean_text, {'entities': [(start_ind, end_ind, label)]}))

    return data
import os
import nltk
import math
import pandas as pd
import numpy as np
import fasttext
import re
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import array
from collections import Counter
from itertools import chain
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# ps = PorterStemmer()

def load_data(file_name):
    """
    :param file_name: a file name, type: str
    return a list of ids, a list of reviews, a list of labels
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name)
    return df["id"], df["text"], df["context"], df["stance_label"], df["impact_label"]


def load_labels(file_name):
    """
    :param file_name: a file name, type: str
    return a list of labels
    """
    return pd.read_csv(file_name)["impact_label"]

def str_to_list(str):
    flag1 = 0
    flag2 = 0
    l = -1
    r = -1
    ans = []
    for i in range(len(str)):
        if l == -1 and str[i] == '\'':
            l = i + 1
            flag1 = 1
        elif l == -1 and str[i] == '"':
            l = i + 1
            flag2 = 1
        if str[i] == '\'' and flag1 == 1:
            if i + 1 == len(str):
                r = i - 1
            elif str[i + 1] == ',' or str[i + 1] == ']':
                r = i - 1
        if str[i] == '"' and flag2 == 1:
            if i + 1 == len(str):
                r = i - 1
            elif str[i + 1] == ',' or str[i + 1] == ']':
                r = i - 1

        if r != -1:
            ans.append(str[l:r])
            l = r = -1
            flag0 = flag1 = 0
    return ans

# Combine Text and Context for trianing dataset
def text_add_context(text, context):
    punctuations = '''['"]'''
    parent_text = []
    text_text = []
    final_text = []
    for i in range(len(context)):
        parent_text_list = str_to_list(context[i])

        # remove'[]'
        no_punct = str()
        for char in str(parent_text_list[-4:-1]):
            if char not in punctuations:
                no_punct = no_punct + char
        final_text.append([str(text[i]) + " " + no_punct])
    return final_text

# Combine Text and Context for testing dataset
def text_add_context_test(text, context):
    punctuations = '''['"]'''
    parent_text = []
    text_text = []
    final_text = []
    for i in range(len(context)):
        parent_text_list = str_to_list(context[i])
        # remove'[]'
        no_punct = str()
        for char in str(parent_text_list[-4:-1]):
            if char not in punctuations:
                no_punct = no_punct + char
        final_text.append([str(text[i]) + " " + no_punct])
    return final_text

# Combine Text,Context and Instance label for trianing dataset
def text_add_context_instance(text, context, instance):
    punctuations = '''['"]'''
    parent_text = []
    text_text = []
    final_text = []
    for i in range(len(context)):
        parent_text_list = str_to_list(context[i])
        # remove'[]'
        no_punct = str()
        for char in str(parent_text_list[-4:-1]):
            if char not in punctuations:
                no_punct = no_punct + char
        final_text.append([instance[i] + " " + str(text[i]) + " " + no_punct])
    return final_text

# Combine Text,Context and Instance label for test dataset
def text_add_context_instance_test(text, context, instance):
    punctuations = '''['"]'''
    parent_text = []
    text_text = []
    final_text = []
    for i in range(len(context)):
        parent_text_list = str_to_list(context[i])
        # remove'[]'
        no_punct = str()
        for char in str(parent_text_list[-4:-1]):
            if char not in punctuations:
                no_punct = no_punct + char
        final_text.append([instance[i] + " " + str(text[i]) + " " + no_punct])
    return final_text


def write_predictions(file_name, id, pred):
    df = pd.DataFrame(zip(id, pred))
    df.columns = ["id", "pred"]
    df.to_csv(file_name, index=False)


def tokenize(text):
    """
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    e.g.
    Input: 'Text mining is to identify useful information.'
    Output: ['Text', 'mining', 'is', 'to', 'identify', 'useful', 'information', '.']
    """
    return nltk.word_tokenize(text)

def precossing_data_fasttext(texts, label):
    final = []
    increased_data = []
    for i in range(len(label)):

        # DownSampling
        if label[i] == 'IMPACTFUL':
            if (i % 2) != 0:
                final.append("__label__" + str(label[i]).lower() + " " + str(texts[i]))
            elif (i % 2) == 0:
                continue
        else:
            final.append("__label__" + str(label[i]).lower() + " " + str(texts[i]))

    return final

def review_to_words(raw_text):
    final = []
    for i in range(len(raw_text)):
        text = str(raw_text[i])
        letters_only = re.sub("[^a-zA-Z]", " ", text)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]
        final.append(meaningful_words)
    return final

def precossing_data_fasttext_test(texts):
    final = []
    for i in range(len(texts)):
        final.append(str(texts[i]))
    return final

def convert_label(label):
    index_label_list=[]
    label_list = label[0]
    for i in range(len(label_list)):
        if str(label_list[i]) == '''['__label__not_impactful']''':
            index_label = 0
        elif str(label_list[i]) == '''['__label__medium_impact']''':
            index_label = 1
        elif str(label_list[i]) == '''['__label__impactful']''':
            index_label = 2
        index_label_list.append(index_label)
    return index_label_list

def convert_label_true(label):
    index_label_list=[]
    for i in range(len(label)):
        if label[i] == 'NOT_IMPACTFUL':
            index_label = 0
        elif label[i] == 'MEDIUM_IMPACT':
            index_label = 1
        elif label[i] == 'IMPACTFUL':
            index_label = 2
        index_label_list.append(index_label)
    return index_label_list

if __name__ == '__main__':
    train_file = "../train.csv"
    vali_file = "../valid.csv"

    # load data
    train_ids, train_texts, train_context, train_stance_label, train_labels = load_data(train_file)
    vali_ids, vali_texts, vali_context, vali_stance_label, vali_labels = load_data(vali_file)

    # Combine Text, Context and Instance label for trianing/Validation dataset
    train_texts_1 = text_add_context_instance(train_texts, train_context, train_stance_label)
    vali_texts_1 = text_add_context_instance(vali_texts, vali_context, vali_stance_label)

    # Precessing the review text and convert to words
    train_words = review_to_words(train_texts_1)
    vali_words = review_to_words(vali_texts_1)

    # DownSampling the 'IMPACTFUL' class
    # and Precessing training words to suittable format for fastText model
    train_text_file = precossing_data_fasttext(train_words, train_labels)

    # Write all corrected formatted training words to text file
    with open('train.txt', 'w') as f:
        for item in train_text_file:
            f.write("%s\n" % item)

    #Training FastText Models
    model = fasttext.train_supervised(input="./train.txt", lr=0.8, label_prefix="__label__", epoch=15, wordNgrams=2,
                                      dim=300, loss='hs')

    # Precessing validation words to correct format for predicting by fastText model
    vali_text_file = precossing_data_fasttext_test(vali_words)
    vali_pred = model.predict(vali_text_file)

    # Convert class label to number
    # ['NOT_IMPACTFUL': 0,'MEDIUM_IMPACT':1,'IMPACTFUL': 2]
    vali_y_pred = convert_label(vali_pred)
    vali_y_true = convert_label_true(vali_labels)

    #Print out the classification Report for the Validation Dataset
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(vali_y_true, vali_y_pred, target_names=target_names))

    ###############
    # Test Datatset
    test_file = "../test.csv"
    test_ids, test_texts, test_context, test_stance_label, test_labels = load_data(test_file)
    test_texts_1 = text_add_context_instance_test(test_texts, test_context, test_stance_label)

    test_words = review_to_words(test_texts_1)
    test_text_file = precossing_data_fasttext_test(test_words)

    predect_label = model.predict(test_text_file)
    predect_output = convert_label(predect_label)

    #predect_output = array(predect_output) - 1
    # Write out the predicted results for text dataset
    write_predictions("Fasttext.csv", test_ids, predect_output)

    #unique, counts = np.unique(predect_output, return_counts=True)
    #dict(zip(unique, counts))

# -*- coding: utf-8 -*-
import re
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

root = "C:/Users/1/James/grctc/GRCTC_Project/Classification/"
write_path = root+"Sequential_Models/word2vector/"
filename = root+"Preprocessing/data/FinalAnnotationsModality_sentences.txt"
googleVecs = "TrainingData_Embeddings"

def reg_pat(match):

    clean_match =match.group(0).replace("<", " ") \
        .replace(">", " ") \
        .replace("/", "") \
        .replace("gate:gateId=", "") \
        .replace("\"", "") \
        .replace("â", "") \
        .replace("€", "") \
        .replace("˜", "") \
        .replace("™", "") \
        .replace("�", "") \
        .replace("œ", "") \
        .replace("œ â", "")

    return clean_match

def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"^,", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower()


def labels2int(labels):
    s = pd.Series(labels)
    f = pd.factorize(s)#,sort=True)
    return f[0]

def clean_data(filepath=""):

    dataset = [str(clean_str(sentence.split(',\'')[1])).split() for sentence in open(filepath, "rb").readlines() if len(sentence.split(',\''))>1]
    labels = [str(sentence.split(',\'')[0]).lower().replace(" ","") for sentence in open(filepath,'rb').readlines() if len(sentence.split(',\''))>1]
    frame = pd.DataFrame({'texts': dataset,'labels': labels})
    frame = frame.sort_values(by='labels')
    label_ids = labels2int(frame['labels'].tolist())
    texts = frame['texts'].tolist()
    frame = pd.DataFrame({'texts': texts,'labels': label_ids})
    frame = frame.reindex(np.random.permutation(frame.index))
    texts = frame['texts'].tolist() ; label_ids = frame['labels'].tolist()
    return texts,label_ids

def get_embeddings(document,model):

    max_words,max_length = model.syn0.shape[0],model.syn0.shape[1]
    arr = np.zeros(max_length, dtype='float32')
    B = []
    for sentence in document:
        A = []
        for word in sentence:
            try:
                emb =model[word]
                A.append(emb)
            except:
                A.append(arr)

        A = np.array(A)
        difference = 100 - len(A)
        A = np.resize(A,(100,max_length))
        #print (A, A.shape)
        B.append(A)
    sequences = np.array(B)
    return sequences

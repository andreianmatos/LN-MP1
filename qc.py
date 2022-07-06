# NATURAL LANGUAGE 2020-2021
# MP1
# Andreia Matos - ist189413

import re
import operator
import os
import sys
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC 
from sklearn.multiclass import OneVsRestClassifier
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 

# splitting labels and questions into different files
def split_file(file):
    with open(os.path.join(sys.path[0], file), "r") as f:
        with open(os.path.join(sys.path[0], file.partition(".")[0]+"-labels.txt"), "w+") as flabels:
            with open(os.path.join(sys.path[0], file.partition(".")[0]+"-questions.txt"), "w+") as fquestions:
                spl_word = ' '
                for line in f:
                    flabels.write(line.partition(spl_word)[0] + '\n')
                    fquestions.write(line.partition(spl_word)[2])  

# text tagging for coarse predictions
def tag_text(text):
    tagged_text = []
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if(hasattr(chunk, 'label')):
                tagged_text.append(chunk.label())
            else:
                for c in chunk[:-1]:
                    if(any(char.isdigit() for char in c)):
                        tagged_text.append("NUM")
                    else:
                        tagged_text.append(c)
    return tagged_text

# text pre procesing for tf idf
def process_text(text, tag):
    processed_text = []
    tags = ["NUM", "PERSON", "GPE", "ORGANIZATION", "GSP","LOCATION","FACILITY"]
    custom_stop_words = ["a", "an", "and", "the", "by","of", "on", "can","all","so","my",
    "doe","t","s","nor","should","shouldn","could","couldn","as", "me", "myself","but", "no","doe"]
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    custom_punctuation = punctuation.replace("-", "")
    if(tag):
        text = tag_text(text)
    else:
        text = word_tokenize(text)
    for w in text:
        if(w in tags):
            processed_text.append(w)
        elif(w not in custom_stop_words):
            processed_text.append(lemmatizer.lemmatize(stemmer.stem(w.lower()))
            .translate(str.maketrans('', '', custom_punctuation)).replace("-", ""))
    return processed_text

def classify_questions(type, train, pred):
    
    # input prediction type: coarse or fine
    pred_type = type

    train_questions = []
    # train questions with labeled words
    train_questions_coarse = []
    # train questions with non-labeled words
    train_questions_fine = []

    train_labels_coarse = []
    train_labels_fine = []

    pred_questions = []
    # prediction questions with labeled words
    pred_questions_coarse = []
    # prediction questions with non-labeled words
    pred_questions_fine = []

    # storing questions and labels from train file
    with open(os.path.join(sys.path[0], train), "r") as tr_data:
        spl_word = ' '
        for line in tr_data:
            question = line.partition(spl_word)[2]
            label = line.partition(spl_word)[0]
            train_questions.append(question)
            train_labels_coarse.append(label.split(":")[0])
            train_labels_fine.append(label.split(":")[1])

    # storing questions from test file
    with open(os.path.join(sys.path[0], pred), "r") as pred_data:
        for question in pred_data:
            pred_questions.append(question)
    
    # text preprocessing for each prediction type
    for q in range(len(train_questions)):
        train_questions_coarse.append(" ".join(process_text(train_questions[q],1)))
        train_questions_fine.append(" ".join(process_text(train_questions[q],0)))

    for q in range(len(pred_questions)):
        pred_questions_coarse.append(" ".join(process_text(pred_questions[q],1)))
        pred_questions_fine.append(" ".join(process_text(pred_questions[q],0)))
    
    # training set
    train_X_coarse = train_questions_coarse 
    train_y_coarse = train_labels_coarse
    
    train_X_fine = train_questions_fine 
    train_y_fine = train_labels_fine

    # testing set
    test_X_coarse = pred_questions_coarse 
    test_X_fine = pred_questions_fine

    # convert text to feature vector with tf idf
    vectorizer = TfidfVectorizer(ngram_range=(1,2))

    X_train_c_tf = vectorizer.fit_transform(train_X_coarse)
    X_test_c_tf = vectorizer.transform(test_X_coarse)

    X_train_f_tf = vectorizer.fit_transform(train_X_fine)
    X_test_f_tf = vectorizer.transform(test_X_fine)


    # train classifier
    svm_model_linear = OneVsRestClassifier(SVC(kernel = 'linear', C = 1))
    svm_model_linear.fit(X_train_c_tf, train_y_coarse)

    # make coarse predictions
    y_pred_coarse = svm_model_linear.predict(X_test_c_tf) 
    # make fine predictions
    svm_model_linear.fit(X_train_f_tf, train_y_fine)
    y_pred_fine = svm_model_linear.predict(X_test_f_tf) 

    # printing results
    if(pred_type == "-coarse"):
        for label in y_pred_coarse:
            print(label)
    else:
        for lc,lf in zip(y_pred_coarse,y_pred_fine):
            print(lc + ":" + lf)

split_file("DEV.txt")
classify_questions(sys.argv[1], sys.argv[2], sys.argv[3])
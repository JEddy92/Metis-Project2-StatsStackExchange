#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicting Cross Validated Stack Exchange Views --
Script for building regression models
Features: reputation, time elapsed, word count, topic tags, tf-idf word values

Created on Fri Jul  7 13:09:20 2017

@author: josepheddy
"""

#==============================================================================
# BLOCK 1: Imports, Data Setup and Pre-processing
#==============================================================================

import pandas as pd
import ast

import numpy as np
import dateutil.parser

import matplotlib.pyplot as plt
import seaborn as sns

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

'''Helper functions for column formatting and feature extraction
'''

def col_to_intcol(df_col):
    new_col = pd.to_numeric(df_col.str.replace('k','000') \
                                  .str.replace('.','') \
                                  .str.replace(',','')) \
                                  .replace(np.nan,0)
    return new_col

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens

def count_words(textstr):
    return len(textstr.split(' '))

def days_before_today(timestr):
    try:
        today = dateutil.parser.parse('2017-07-08')
        post_day = dateutil.parser.parse(timestr).replace(tzinfo=None)
        delta = (post_day - today).days
        return -delta
    except:
        return np.nan
    
'''Dataframe setup and preprocessing
'''    
    
df = pd.read_csv('SE_Questions_pgs1-2000.csv')
df = df.drop(df.columns[0], axis=1)

df['views'] = col_to_intcol(df['views'])
df['reputation'] = col_to_intcol(df['reputation'])
df['favs'] = df['favs'].replace(np.nan,0)
df['votes'] = df['votes'].replace(np.nan,0)
df[df['votes'] < 0] = 0 #zero out down votes
  
#convert list formatted strings to actual lists 
df['topic_tags'] = df['topic_tags'].replace(0,'[]')
df['topic_tags'] = df['topic_tags'].apply(ast.literal_eval) 

df['days_elapsed'] = df['time'].apply(days_before_today)
df.dropna(inplace=True)
  
df['word_count'] = df['text'].apply(count_words)
                        
#log transform right skewed columns
df[['views','reputation','favs','votes']] = \
    np.log(df[['views','reputation','favs','votes']] + 1) 
                           
df['text'] = df['text'].str.replace(r'[\W_]+',' ')
df['text'] = df['text'].replace(np.nan,'')

#nice list -> dummies conversion, store dummies
tag_dummies = df['topic_tags'].str.join(sep='*').str.get_dummies(sep='*') 

#%%
#==============================================================================
# BLOCK 2: Basic model creation. Reputation, time, word count, topic tags
#==============================================================================

X_basic = pd.concat([df[['reputation','days_elapsed','word_count']],tag_dummies], axis=1)
y_basic = df['views']

#66% train set, 33% test set
X_basic_train, X_basic_test, y_basic_train, y_basic_test = train_test_split( \
                                                                            X_basic, \
                                                                            y_basic, \
                                                                            test_size=0.33, \
                                                                            random_state=30
                                                                            )       

#Simple regression. Easy to see that the model is highly unstable - likely due to
#use of many sparse features. 
model_basic = LinearRegression()
model_basic.fit(X_basic_train,y_basic_train)
print('Basic Linear Model Train score: %.5f' % \
      model_basic.score(X_basic_train,y_basic_train))
print('Basic Linear Model Test score: %.5f' % \
      model_basic.score(X_basic_test,y_basic_test))

#Want to use regularization, so should standardize the non categorical features.
scale = preprocessing.StandardScaler()
X_basic_train[['reputation','days_elapsed','word_count']] = \
             scale.fit_transform(X_basic_train[['reputation','days_elapsed','word_count']])
X_basic_test[['reputation','days_elapsed','word_count']] = \
            scale.transform(X_basic_test[['reputation','days_elapsed','word_count']])

#Ridge regularized regression - vast improvement in model stability.
#Alpha = 10 can get .~.44 test R^2
model_basic_ridge = Ridge(alpha=10)
model_basic_ridge.fit(X_basic_train,y_basic_train)
print('Basic Ridge (alpha=10) Model Train score: %.5f' % \
      model_basic_ridge.score(X_basic_train,y_basic_train))
print('Basic Ridge (alpha=10) Model Test score: %.5f' % \
      model_basic_ridge.score(X_basic_test,y_basic_test))

#Plotting the alpha parameter selection based on train/test MSE.
#Supporting argument for choice of alpha=10
train_mse = []
test_mse = []
alphas = [10**i for i in range(-5,5)]
for val in alphas:
        ridge_alph = Ridge(alpha=val)
        ridge_alph.fit(X_basic_train,y_basic_train)
        train_mse.append(mean_squared_error(ridge_alph.predict(X_basic_train),y_basic_train))
        test_mse.append(mean_squared_error(ridge_alph.predict(X_basic_test),y_basic_test))

plt.figure(figsize=(8,6))
plt.plot(alphas, train_mse, color='green', label='train')
plt.plot(alphas, test_mse, color='red', label='test')
plt.xscale('log')
plt.ylabel('mean squared error')
plt.xlabel('Ridge alpha value')
plt.legend(loc='upper left')
plt.title('Ridge Regression Train/Test MSEs vs. Regularization Strength')
plt.show()

#%%
#==============================================================================
# BLOCK 3: Final model creation. Reputation, time, word count, topic tags and tf-idfs
#==============================================================================

'''Model building process including tf-idfs extraction
'''

X = pd.concat([df[['reputation','days_elapsed','word_count','text']],tag_dummies], axis=1)
y = df['views']

#66% train set, 33% test set
X_train, X_test, y_train, y_test = train_test_split( \
                                                    X, \
                                                    y, \
                                                    test_size=0.33, \
                                                    random_state=20
                                                    )

#Using regularization, so want to standardize continuous features
scale = preprocessing.StandardScaler()
X_train[['reputation','days_elapsed','word_count']] = \
       scale.fit_transform(X_train[['reputation','days_elapsed','word_count']])
X_test[['reputation','days_elapsed','word_count']] = \
      scale.transform(X_test[['reputation','days_elapsed','word_count']])

#Process train tf-idfs and add these features to the train set
tfidf = TfidfVectorizer(tokenizer=tokenize, min_df = 50, stop_words='english')
tfs_train = tfidf.fit_transform(X_train['text']).toarray()
X_train = np.hstack((X_train.drop('text',axis=1).values,tfs_train))

#Use same tf-idf processor to extract test tf-idfs and add to test set
tfs_test = tfidf.transform(X_test['text']).toarray()
X_test = np.hstack((X_test.drop('text',axis=1).values,tfs_test))

#Ridge regression w alpha=15 is test r^2 ~.48 (better than above basic model). 
# So using tf-idf features does seem to modestly improve the quality of the model
model = Ridge(alpha=15)
model.fit(X_train, y_train)
print('Training score: %.5f' % model.score(X_train, y_train))
print('Test score: %.5f' % model.score(X_test, y_test))

'''Some prediction / residual diagnostics plotting.

   Actuals vs. predicted suggests we're starting to do worse at the upper
   end of the range. Predicted vs. residuals does not seem to indicate
   too severe a level of heteroskedasticity though. Could dig into this more.
'''

y_pred = model.predict(X_train) 
resids = y_train - y_pred
df_act_resid = pd.DataFrame({'y_actual': y_train, 'y_pred': y_pred, 'resids': resids})

ax = sns.jointplot('y_pred', 'y_actual', kind='regplot', data=df_act_resid, scatter_kws={'alpha':0.3})
plt.xlabel('Predicted log(views)')
plt.ylabel('Actual log(views)')
plt.show()

sns.jointplot('y_pred', 'resids', kind='regplot', data=df_act_resid, scatter_kws={'alpha':0.3})
plt.xlabel('Predicted log(views)')
plt.ylabel('Residuals')
plt.show()

'''Taking a look at feature coefficients to see if any interesting findings.

   Some pretty cool stuff here! It seems that words associated with understanding,
   explaining, or interpreting tend to promote viewership, while words associated 
   with more immediate problem solving like 'want' or 'problem' tend to decrease it.
   
   The top few tag coeffs correspond to careers, paradox, intuition, correlation-matrix,
   and regression-strategies (so meta!).
'''

#gather together all the feature names
tags = [tag + '_TAG' for tag in tag_dummies.columns]
tfidf_words = [word + '_tf-idf' for word in tfidf.get_feature_names()]
coef_names = ['reputation','days_elapsed','word_count'] + tags + tfidf_words 

#create coefs df, sort by absolute value of coef and look at top ones
df_coefs = pd.DataFrame({'feature':coef_names, 'coef':model.coef_})
df_coefs = df_coefs.reindex(df_coefs['coef'].abs().order(ascending=False).index)
print(df_coefs[:15])

#do the same but breaking out tags and tf-idfs separately
df_tag_coefs = df_coefs[df_coefs['feature'].str.contains('_TAG')]
df_tfidf_coefs = df_coefs[df_coefs['feature'].str.contains('_tf-idf')]

df_tag_coefs = df_tag_coefs.reindex(df_tag_coefs['coef'].abs().order(ascending=False).index)
print(df_tag_coefs[:15])

df_tfidf_coefs = df_tfidf_coefs.reindex(df_tfidf_coefs['coef'].abs().order(ascending=False).index)
print(df_tfidf_coefs[:15])


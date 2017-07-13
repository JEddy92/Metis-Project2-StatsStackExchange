#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scraping Cross Validated Stack Exchange Questions --
Script for crawling over question pages to extra metadata and text

Created on Thu Jul  6 14:26:53 2017

@author: josepheddy
"""

import requests
from bs4 import BeautifulSoup
import time

import dateutil.parser
import pandas as pd

'''Helper function for page parsing -- convert url to soup, and make sure you 
   are not getting a blocked status code / stop hitting the site for a while if you do
'''
def url_to_soup(url):
    response = requests.get(url)
    while (response.status_code == 429):
            print('Too many requests, stop making StackExchange sad. 5 Minute break.')
            time.sleep(300)
            response = requests.get(url)  
    
    return BeautifulSoup(response.text,"html5lib") 

'''Helper function for pulling an individual question's data from a corresponding soup
'''
def pull_question_data(soup):
    
        raw_title = soup.find('title').text                    
        title = raw_title.rsplit('-',1)[0] \
                         .split('-',1)[-1].strip()
        
        sidebar = soup.find(id="qinfo")
        if (sidebar):
            sidebar = sidebar.find_all('td')
            raw_time = sidebar[1].find(class_ ='label-key')['title']
            time = dateutil.parser.parse(raw_time)
            views = sidebar[3].find(class_ ='label-key').text.split()[0]
        else:
            time, views = None, None
                         
        raw_topic_tags = soup.find(class_='post-taglist')
        if (raw_topic_tags):
            topic_tags = raw_topic_tags.text.strip().split(' ')
        else:
            topic_tags = []
        
        raw_text = soup.find(class_='post-text')
        if (raw_text):
            text = raw_text.text
        else:
            text = None
        
        raw_reputation = soup.find(class_='reputation-score')
        if (raw_reputation):
            reputation = raw_reputation.text
        else:
            reputation = None
                        
        raw_votes = soup.find(class_='vote-count-post')
        if (raw_votes):
            votes = raw_votes.text 
        else:
            votes = None
            
        raw_favs = soup.find(class_='favoritecount')      
        if (raw_favs):
            favs = raw_favs.text 
        else:
            favs = None
            
        headers = ['title','time','views','topic_tags',
                   'text','reputation','votes','favs']
        
        #return collected data in dictionary format
        question_dict = dict(zip(headers, [title,time,views,topic_tags,text,reputation,votes,favs]))
        return question_dict
 
'''Function for pulling all question data from a soup corresponding to a page of questions --
   Grab all the question urls and pull data from each.
'''          
def crawl_page_questions(soup):
    
    question_dicts = [] 
    search_soup = soup
    question_htmls = search_soup.find_all(class_='question-summary')
    
    for question_html in question_htmls:
        question_url = ('https://stats.stackexchange.com' \
                        + question_html.find(class_='question-hyperlink')['href']) 
        question_soup = url_to_soup(question_url) 
        question_dict = pull_question_data(question_soup)
        question_dicts.append(question_dict)        
        
    return question_dicts

'''Go through a bunch of question pages and extract their data. Take breaks in fixed
   intervals so that you don't overload the site with requests.
   
   Set a page count at which to save the scraped data as a csv and then dump it.
   Page count set to 250 here.
'''   
data = []
start, end = 1,250
bench = start
for i in range(start,end+1):
    search_url = 'https://stats.stackexchange.com/questions?page=%d&sort=newest' %i  
    search_soup = url_to_soup(search_url)
    data.extend(crawl_page_questions(search_soup))
        
    if (i % 3 == 0):
        print('Sleeping For 2 minutes')
        time.sleep(120)
        
    if (i % 250 == 0):
        print('Saving a results csv')
        df_data = pd.DataFrame(data)
        df_data.to_csv('SE_Questions_pgs%d-%d.csv' %(bench,i))
        bench = i
        data = []

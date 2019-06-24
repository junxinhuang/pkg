import nltk
import pandas as pd
import os 
import re
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
from textblob import TextBlob
from evaluate_csv import KW
import sys


#Lemmatize
def split_into_lemmas(message):
    message = str(message).lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma     
    return ' '.join([word.lemma for word in words])


def remove_stop(s):
    word_tokens = word_tokenize(s)
    ll = [w for w in word_tokens if not w in stop_words]
    return ' '.join(ll)

def col_remove(column):
    return column.apply(remove_stop)
    
def clean(df):
    df['lemmatize'] = df['Keywords'].apply(split_into_lemmas)
    df['lemmatize_nonstop']=col_remove(df['lemmatize'])    
    df.rename(columns={'lemmatize_nonstop':'Query'}, inplace=True)
    df['volume']=df['volume'].str.replace(',','').astype('int')
    return df

#pos tagging
def simple_wordcloud(df):
    d = {}
    for a, x in df.values:
        d[a] = x
    
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=d)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def tagging(s):
    tokens = nltk.word_tokenize(s)
    tagged = nltk.pos_tag(tokens)
    return tagged

# single words  return a df saying words and type

def singleword(df):   # see the distribution of the single type
    coblist=[] # put all words together    
    taglist=(df['Query'].apply(tagging)).tolist()
    for i in taglist:  # three tuples are a list, list level
        for j in i:    # at single tuple level
            coblist.append(j)
    df_1 = pd.DataFrame(coblist, columns=['Query','type'])  
    return df_1

def singletype(df):   # see the distribution of the single type
    df_1 = singleword(df)
    return df_1.groupby('type').count().sort_values('Query').reset_index()      



def singleword_msearch(df):   # see the distribution of the single type
    coblist=[] # put all words together
    taglist=(df['Query'].apply(tagging)).tolist()
    for i in range(len(taglist)):  # three tuples are a list, list level
        for j in taglist[i]:    # at single tuple level
            coblist.append([j[0],j[1],df['volume'][i]])
    df_1 = pd.DataFrame(coblist, columns=['Query','type','volume'])  
    return df_1



def singletype_msearch(df):   # see the distribution of the single type
    df_1 = singleword_msearch(df)
    return df_1[['type','volume']].groupby('type').sum().sort_values('volume').reset_index()      


def typeinwords(df):
    cobtlist=[] # put all tuples list together
    taglist=(df['Query'].apply(tagging)).tolist()
    for i in taglist:  # three tuples are a list, list level
        tuplist=[]
        for j in i:
            tuplist.append(j[1])
        cobtlist.append(tuplist)    
    cobtlist1 = [', '.join(l) for l in cobtlist]
    return cobtlist1


def one_hot_encoding(df):
    columns = singletype(df)['type'].tolist()
    df_col = pd.DataFrame(columns)
    vv = []
    for i in typeinwords(df):
        v = []
        for j in i.split(', '):
            l=list(j == df_col.iloc[:,0])
            v.append([int(i) for i in l])
        vs = sum([np.array(i) for i in v])
        vv.append(vs)
    return pd.DataFrame(vv,columns=columns)

vl=['VB','VBD','VBG','VBN','VBP','VBZ']
n=['NN','NNS','NNP','NNPS']
jj=['JJ','JJR','JJS']
#allwords=df_ohe.columns

# below is the condition words could function
def singleword_condition(df, word):
    taglist=(df['Query'].apply(tagging)).tolist()
    mid_list=[]
    spe_list=[]
    for i in taglist:
        for j in i:
            if j[0] == word:
                mid_list.append(i)
    for i in mid_list:  # three tuples are a list, list level
        for j in i:     # at single tuple level
            spe_list.append(j)
    df_1 = pd.DataFrame(spe_list, columns=['Query','type'])  
    df_2 = df_1[df_1['Query']!=word]
    return df_2.reset_index(drop=True)



def singleword_condition_msearch(df, word):
    taglist=(df['Query'].apply(tagging)).tolist()
    mid_ind=[]
    spe_list=[]
    for i in range(len(taglist)):
        for j in taglist[i]:
            if j[0] == word:
                mid_ind.append(i)
    for i in mid_ind:  # three tuples are a list, list level
        for j in taglist[i]:     # at single tuple level
            spe_list.append([j[0],j[1],df['volume'][i]])
            
    df_1 = pd.DataFrame(spe_list, columns=['Query','type','volume'])  
    df_2 = df_1[df_1['Query']!=word]
    return df_2.reset_index(drop=True)




def uglywordcloud_condition(df,typelist,word):   # requires your dataframe;  type of words list, such as n,jj,vl   ; and the 'word'
                                                 # so that it will return the type of words corresponding with the 'word'
    spe_list=singleword_condition(df, word)
    df_singleword=singleword(spe_list)
    df_singleword_f=df_singleword[[i in typelist for i in df_singleword['type']]]
    single_gp=df_singleword_f.groupby('Query').count().sort_values('type',ascending= False).reset_index()
    simple_wordcloud(single_gp)# wordf cloud of the most popular vb
    print('Among words corresponding to {}, the top 10 words in {} are :{} '.format(word, typelist,single_gp['Query'][:10].tolist()))
    print()
    print('Among words corresponding to {}, the percentage of {} is {}'.format(word,typelist,df_singleword_f.shape[0]/df_singleword.shape[0]))
    return single_gp



def uglywordcloud_condition_msearch(df,typelist,word):   # requires your dataframe;  type of words list, such as n,jj,vl   ; and the 'word'
                                                 # so that it will return the type of words corresponding with the 'word'

    spe_list=singleword_condition_msearch(df, word)
    df_singleword=singleword_msearch(spe_list)
    df_singleword_f=df_singleword[[i in typelist for i in df_singleword['type']]]
    single_gp=df_singleword_f[['Query','volume']].groupby('Query').sum().sort_values('volume').reset_index()   
    simple_wordcloud(single_gp)# wordf cloud of the most popular vb
    print('Among words corresponding to {}, the top 10 words in {} are :{} '.format(word, typelist,single_gp['Query'][:10].tolist()))
    print()
    print('Among words corresponding to {}, the percentage of {} is {}'.format(word,typelist,df_singleword_f.shape[0]/df_singleword.shape[0]))
    return single_gp



if __name__=='__main__':
    kw = KW(sys.argv[1],sys.argv[2])
    df = KW.getKWfile(kw)

    df=clean(df)
    # df.to_csv('df.csv')
    # df_ohe = one_hot_encoding(df) 
    # df_ohe.to_csv('df_ohe.csv')
    uglywordcloud_condition(df,n,'pregnancy')
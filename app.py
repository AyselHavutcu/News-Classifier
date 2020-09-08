import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import punkt
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_renderer
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import re
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
"""
1. Importing inputs

"""
#Trained Model
#The best performing model is the SVM. We'll use it in the app.
path_model = "C:/Users/HP/Desktop/stajprojesi/usedpickles/best_svc.pickle"

# SVM
with open(path_model, 'rb') as data:
    svc_model = pickle.load(data)



#TF-IDF object
path_tfidf = "C:/Users/HP/Desktop/stajprojesi/usedpickles/tfidf.pickle"

with open(path_tfidf, 'rb') as data:
    tfidf = pickle.load(data)

#Category mapping dictionary

category_codes = {
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sport': 3,
    'tech': 4,
    'other':5
}

#Web Scraping Functions
#Dailymail

def get_news_dailymail():

    # url definition
    url = "https://www.dailymail.co.uk"


    # Request
    r1 = requests.get(url)

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification
    coverpage_news = soup1.find_all('h2', class_='linkro-darkred')

    number_of_articles = 5

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    for n in np.arange(0, number_of_articles):

        # Getting the link of the article
        link = url + coverpage_news[n].find('a')['href']
        list_links.append(link)

        # Getting the title
        title = coverpage_news[n].find('a').get_text()
        list_titles.append(title)

        # Reading the content (it is divided in paragraphs)
        article = requests.get(link)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html5lib')
        body = soup_article.find_all('p', class_='mol-para-with-font')

        # Unifying the paragraphs
        list_paragraphs = []
        for p in np.arange(0, len(body)):
            paragraph = body[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)

        # Removing special characters
        final_article = re.sub("\\xa0", "", final_article)

        news_contents.append(final_article)

    # df_features
    df_features = pd.DataFrame(
         {'Content': news_contents  
        })

    # df_show_info
    df_show_info = pd.DataFrame(
        {'Article Title': list_titles,
         'Article Link': list_links,
          'Newspaper': 'Daily Mail'})
    
    return (df_features, df_show_info)

def get_news_skynews():
    
    # url definition
    url = "https://news.sky.com/us"

    # Request
    r1 = requests.get(url)

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification
    coverpage_news = soup1.find_all('h3', class_="sdc-site-tile__headline")

    number_of_articles = 5

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    for n in np.arange(0, number_of_articles):

        # Getting the link of the article
        link = "https://news.sky.com" + coverpage_news[n].find('a', class_='sdc-site-tile__headline-link')['href']
        list_links.append(link)

        # Getting the title
        title = coverpage_news[n].find('a').find('span').get_text()
        list_titles.append(title)

        # Reading the content (it is divided in paragraphs)
        article = requests.get(link)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html5lib')
        body = soup_article.find_all('div', class_='sdc-article-body sdc-article-body--story sdc-article-body--lead')
        x = body[0].find_all('p')

        # Unifying the paragraphs
        list_paragraphs = []
        for p in np.arange(0, len(x)):
            paragraph = x[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)

        news_contents.append(final_article)

    # df_features
    df_features = pd.DataFrame(
         {'Content': news_contents 
        })

    # df_show_info
    df_show_info = pd.DataFrame(
        {'Article Title': list_titles,
         'Article Link': list_links,
         'Newspaper': 'Sky News'})
    
    return (df_features, df_show_info)
#Hurriyet
def get_news_hurriyet():
   
    #define url
    url="https://www.hurriyetdailynews.com/"

    #List of news:
    # Request
    r1 = requests.get(url)

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification
    coverpage_news = soup1.find_all('div', class_='news')

    #print(len(coverpage_news))
    #Let's extract the text from the articles:
    number_of_articles = 5
    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []
    for n in np.arange(0, number_of_articles):

        # We need to ignore "live" pages since they are not articles
        if "live" in coverpage_news[n].find('a')['href']:  
            continue

        # Getting the link of the article
        link = coverpage_news[n].find('a')['href']
        list_links.append(link)

        # Getting the title
        title = coverpage_news[n].find('a').get_text()
        list_titles.append(title)

        # Reading the content (it is divided in paragraphs)
        article = requests.get("https://www.hurriyetdailynews.com/" + link)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html5lib')
        body = soup_article.find_all('div', class_='content')
        if(len(body) != 0):
            x = body[0].find_all('p')
        else:
            print("there is no article content")

        # Unifying the paragraphs
        list_paragraphs = []
        for p in np.arange(0, len(x)):
            paragraph = x[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)

        news_contents.append(final_article)


    # df_features
    df_features = pd.DataFrame(
         {'Content': news_contents 
        })

    # df_show_info
    df_show_info = pd.DataFrame(
        {'Article Title': list_titles,
         'Article Link': list_links,
        'Newspaper': 'Hurriyet'})
   

    return (df_features, df_show_info)

#sabah
def get_news_sabah():

    #url
    url="https://www.dailysabah.com/"
    # Request
    r1 = requests.get(url)

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')


    # News identification
    coverpage_news = soup1.find_all('div', class_='widget_content')

    #Let's extract the text from the articles:
    number_of_articles = 5
    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []
    for n in np.arange(0, number_of_articles):

        # We need to ignore "live" pages since they are not articles
        if "live" in coverpage_news[n].find('a')['href']:  
            continue

        # Getting the link of the article
        link = coverpage_news[n].find('a')['href']
        list_links.append(link)

        # Getting the title
        title = coverpage_news[n].find('a').get_text()
        list_titles.append(title)

        # Reading the content (it is divided in paragraphs)
        article = requests.get(link)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html5lib')
        body = soup_article.find_all('div', class_='article_body')
        if(len(body) != 0):
            x = body[0].find_all('p')
        else:
            print("there is no article content")

        # Unifying the paragraphs
        list_paragraphs = []
        for p in np.arange(0, len(x)):
            paragraph = x[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)

        news_contents.append(final_article)
    # df_features
    df_features = pd.DataFrame(
         {'Content': news_contents 
        })

    # df_show_info
    df_show_info = pd.DataFrame(
        {'Article Title': list_titles,
         'Article Link': list_links,
         'Newspaper':'Sabah'})
    
    return (df_features,df_show_info)

 



#Feature Engineering Functions
punctuation_signs = list("?:!.,;")
stop_words = list(stopwords.words('english'))

def create_features_from_df(df):
    
    df['Content_Parsed_1'] = df['Content'].str.replace("\r", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("    ", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('"', '')
    
    df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()
    
    df['Content_Parsed_3'] = df['Content_Parsed_2']
    for punct_sign in punctuation_signs:
        df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')
        
    df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")
    
    wordnet_lemmatizer = WordNetLemmatizer()
    nrows = len(df)
    lemmatized_text_list = []
    for row in range(0, nrows):

        # Create an empty list containing lemmatized words
        lemmatized_list = []
        # Save the text and its words into an object
        text = df.loc[row]['Content_Parsed_4']
        text_words = text.split(" ")
        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        # Join the list
        lemmatized_text = " ".join(lemmatized_list)
        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)
    
    df['Content_Parsed_5'] = lemmatized_text_list
    
    df['Content_Parsed_6'] = df['Content_Parsed_5']
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_stopword, '')
        
    df = df['Content_Parsed_6']
    df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})
    
    # TF-IDF
    features = tfidf.transform(df).toarray()
    
    return features


def get_category_name(category_id):
    for category, id_ in category_codes.items():    
        if id_ == category_id:
            return category
        
# Prediction Functions
def predict_from_features(features):
        
    # Obtain the highest probability of the predictions for each article
    predictions_proba = svc_model.predict_proba(features).max(axis=1)    
    
    # Predict using the input model
    predictions_pre = svc_model.predict(features)

    # Replace prediction with 6 if associated cond. probability less than threshold
    predictions = []

    for prob, cat in zip(predictions_proba, predictions_pre):
        if prob > .65:
            predictions.append(cat)
        else:
            predictions.append(5)

    # Return result
    categories = [get_category_name(x) for x in predictions]
    
    return categories

def complete_df(df, categories):
    df['Prediction'] = categories
    return df

# Dash App

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Colors
colors = {
    'background': '#ECECEC',  
    'text': '#696969',
    'titles': '#599ACF',
    'blocks': '#F7F7F7',
    'graph_background': '#F7F7F7',
    'banner': '#C3DCF2'

}

# Markdown text
markdown_text1 = '''

This application gathers the latest news from the newspapers **Daily Mail**, **The Guardian** and **Hurriyet(Turkish News Paper)**, predicts their category between **Politics**, **Business**, **Entertainment**, **Sport**, **Tech** and **Other** and then shows a summary.

The scraped news are converted into a numeric feature vector with *TF-IDF vectorization*. Then, a *Support Vector Classifier* is applied to predict each category.

This app is meant for didactic purposes.

Please enter which newspapers would you like to scrape news off and press the **Scrape** button.

'''

markdown_text2 = '''

 Created by Aysel Havutcu.

 *Disclaimer: this app is not under periodic maintenance. A live web-scraping process is carried out every time you run the app, so there may be some crashes due to the failing status of some requests.*

'''



app.layout = html.Div(style={'backgroundColor':colors['background']}, children=[
    
    # Space before title
    html.H1(children=' ',
            style={'padding': '10px'}
           ),
    
    # Title
    html.Div(
        [
            html.H3(children='News Classification App',
                    style={"margin-bottom": "0px"}
                   ),
            html.H6(children='A Machine Learning based app')
        ],
        style={
            'textAlign': 'center',
            'color': colors['text'],
            #'padding': '0px',
            'backgroundColor': colors['background']
              },
        className='banner',
            ),
    

    # Space after title
    html.H1(children=' ',
            style={'padding': '1px'}),


    # Text boxes
    html.Div(
        [
            html.Div(
                [
                    html.H6(children='What does this app do?',
                            style={'color':colors['titles']}),
                    
                    html.Div(
                        [dcc.Markdown(children=markdown_text1),],
                        style={'font-size': '12px',
                               'color': colors['text']}),
                                        
                    html.Div(
                        [
                            dcc.Dropdown(
                                options=[
                                    {'label': 'Daily Mail', 'value': 'DLM'},
                                    {'label': 'Sky News', 'value': 'SKN'},
                                    {'label': 'Hurriyet', 'value': 'HRT'},
                                    {'label' :'Sabah','value':'SBH'}
                                        ],
                                value=['DLM'],
                                multi=True,
                                id='checklist'),
                        ],
                        style={'font-size': '12px',
                               'margin-top': '25px'}),
                    
                    html.Div([
                        html.Button('Scrape', 
                                    id='submit', 
                                    type='submit', 
                                    style={'color': colors['blocks'],
                                           'background-color': colors['titles'],
                                           'border': 'None'})],
                        style={'textAlign': 'center',
                               'padding': '20px',
                               "margin-bottom": "0px",
                               'color': colors['titles']}),
            
                    dcc.Loading(id="loading-1", children=[html.Div(id="loading-output-1")], type="circle"),
                    
                    html.Hr(),
                    html.H6(children='Headlines',
                            style={'color': colors['titles']}),

                    # Headlines
                    html.A(id="textarea1a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea1b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea2a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea2b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea3a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea3b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea4a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea4b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea5a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea5b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea6a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea6b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea7a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea7b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea8a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea8b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea9a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea9b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea10a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea10b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea11a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea11b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea12a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea12b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea13a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea13b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea14a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea14b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea15a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea15b", style={'color': colors['text'], 'font-size': '11px'})
                                                            
                ],
                     style={'backgroundColor': colors['blocks'],
                            'padding': '20px',
                            'border-radius': '5px',
                            'box-shadow': '1px 1px 1px #9D9D9D'},
                     className='one-half column'),
            
            html.Div(
                [
                    html.H6("Graphic summary",
                            style={'color': colors['titles']}),

                    html.Div([
                         dcc.Graph(id='graph1', style={'height': '300px'})
                         ],
                         style={'backgroundColor': colors['blocks'],
                                'padding': '20px'}
                    ),
                    
                    html.Div([
                         dcc.Graph(id='graph2', style={'height': '300px'})
                         ],
                         style={'backgroundColor': colors['blocks'],
                                'padding': '20px'}
                    )
                ],
                     style={'backgroundColor': colors['blocks'],
                            'padding': '20px',
                            'border-radius': '5px',
                            'box-shadow': '1px 1px 1px #9D9D9D'},
                     className='one-half column')

        ],
        className="row flex-display",
        style={'padding': '20px',
               'margin-bottom': '0px'}
    ),
    
        
    # Space
    html.H1(id='space2', children=' '),
        
    
    # Final paragraph
    html.Div(
            [dcc.Markdown(children=markdown_text2),],
            style={'font-size': '12px',
                   'color': colors['text']}),

    
    # Hidden div inside the app that stores the intermediate value
    html.Div(id='intermediate-value', style={'display': 'none'})
    

])


@app.callback(
    [
    Output('intermediate-value', 'children'),
    Output('loading-1', 'children')
    ],
    [Input('submit', 'n_clicks')],
    [State('checklist', 'value')])
def scrape_and_predict(n_clicks, values):
            
    df_features = pd.DataFrame()
    df_show_info = pd.DataFrame()
    
    if 'DLM' in values:
        # Get the scraped dataframes
        df_features = df_features.append(get_news_dailymail()[0])
        df_show_info = df_show_info.append(get_news_dailymail()[1])
    
    if 'SKN' in values:
        df_features = df_features.append(get_news_skynews()[0])
        df_show_info = df_show_info.append(get_news_skynews()[1])
        
    if 'HRT' in values:
        df_features = df_features.append(get_news_hurriyet()[0])
        df_show_info = df_show_info.append(get_news_hurriyet()[1])
        
    if 'SBH' in values:
        df_features = df_features.append(get_news_sabah()[0])
        df_show_info = df_show_info.append(get_news_sabah()[1])


    df_features = df_features.reset_index().drop('index', axis=1)
    
    # Create features
    features = create_features_from_df(df_features)
    # Predict
    predictions = predict_from_features(features)
    # Put into dataset
    df = complete_df(df_show_info, predictions)
    # df.to_csv('Tableau Teaser/df_tableau.csv', sep='^')  # export to csv to work out an example in Tableau
    
    return df.to_json(date_format='iso', orient='split'), ' '

@app.callback(
    Output('graph1', 'figure'),
    [Input('intermediate-value', 'children')])
def update_barchart(jsonified_df):
    
    df = pd.read_json(jsonified_df, orient='split')
    
    # Create a summary df
    df_sum = df.groupby(['Newspaper', 'Prediction']).count()['Article Title']

    # Create x and y arrays for the bar plot for every newspaper
    if 'Daily Mail' in df_sum.index:
    
        df_sum_epe = df_sum['Daily Mail']
        x_epe = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_epe = [[df_sum_epe['politics'] if 'politics' in df_sum_epe.index else 0][0],
                [df_sum_epe['business'] if 'business' in df_sum_epe.index else 0][0],
                [df_sum_epe['entertainment'] if 'entertainment' in df_sum_epe.index else 0][0],
                [df_sum_epe['sport'] if 'sport' in df_sum_epe.index else 0][0],
                [df_sum_epe['tech'] if 'tech' in df_sum_epe.index else 0][0],
                [df_sum_epe['other'] if 'other' in df_sum_epe.index else 0][0]]   
    else:
        x_epe = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_epe = [0,0,0,0,0,0]
    
    if 'Sky News' in df_sum.index:
        
        df_sum_thg = df_sum['Sky News']
        x_thg = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_thg = [[df_sum_thg['politics'] if 'politics' in df_sum_thg.index else 0][0],
                [df_sum_thg['business'] if 'business' in df_sum_thg.index else 0][0],
                [df_sum_thg['entertainment'] if 'entertainment' in df_sum_thg.index else 0][0],
                [df_sum_thg['sport'] if 'sport' in df_sum_thg.index else 0][0],
                [df_sum_thg['tech'] if 'tech' in df_sum_thg.index else 0][0],
                [df_sum_thg['other'] if 'other' in df_sum_thg.index else 0][0]]   
    else:
        x_thg = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_thg = [0,0,0,0,0,0]

    if 'Hurriyet' in df_sum.index:
    
        df_sum_skn = df_sum['Hurriyet']
        x_skn = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_skn = [[df_sum_skn['politics'] if 'politics' in df_sum_skn.index else 0][0],
                [df_sum_skn['business'] if 'business' in df_sum_skn.index else 0][0],
                [df_sum_skn['entertainment'] if 'entertainment' in df_sum_skn.index else 0][0],
                [df_sum_skn['sport'] if 'sport' in df_sum_skn.index else 0][0],
                [df_sum_skn['tech'] if 'tech' in df_sum_skn.index else 0][0],
                [df_sum_skn['other'] if 'other' in df_sum_skn.index else 0][0]]   

    else:
        x_skn = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_skn = [0,0,0,0,0,0]
        
    if 'Sabah' in df_sum.index:
    
        df_sum_sbh = df_sum['Sabah']
        x_sbh = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_sbh = [[df_sum_sbh['politics'] if 'politics' in df_sum_sbh.index else 0][0],
                [df_sum_sbh['business'] if 'business' in df_sum_sbh.index else 0][0],
                [df_sum_sbh['entertainment'] if 'entertainment' in df_sum_sbh.index else 0][0],
                [df_sum_sbh['sport'] if 'sport' in df_sum_sbh.index else 0][0],
                [df_sum_sbh['tech'] if 'tech' in df_sum_sbh.index else 0][0],
                [df_sum_sbh['other'] if 'other' in df_sum_sbh.index else 0][0]]   

    else:
        x_sbh = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_sbh = [0,0,0,0,0,0]

    # Create plotly figure
    figure = {
        'data': [
            {'x': x_epe, 'y':y_epe, 'type': 'bar', 'name': 'Daily Mail', 'marker': {'color': 'rgb(62, 137, 195)'}},
            {'x': x_thg, 'y':y_thg, 'type': 'bar', 'name': 'Sky News', 'marker': {'color': 'rgb(167, 203, 232)'}},
            {'x': x_skn, 'y':y_skn, 'type': 'bar', 'name': 'Hurriyet', 'marker': {'color': 'rgb(197, 223, 242)'}},
            {'x': x_sbh, 'y':y_sbh, 'type': 'bar', 'name': 'Sabah', 'marker': {'color': 'rgb(197, 223, 242)'}}
        ],
        'layout': {
            'title': 'Number of news articles by newspaper',
            'plot_bgcolor': colors['graph_background'],
            'paper_bgcolor': colors['graph_background'],
            'font': {
                    'color': colors['text'],
                    'size': '10'
            },
            'barmode': 'stack'
            
        }   
    }

    return figure

@app.callback(
    Output('graph2', 'figure'),
    [Input('intermediate-value', 'children')])
def update_piechart(jsonified_df):
    
    df = pd.read_json(jsonified_df, orient='split')
    
    # Create a summary df
    df_sum = df['Prediction'].value_counts()

    # Create x and y arrays for the bar plot
    x = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
    y = [[df_sum['politics'] if 'politics' in df_sum.index else 0][0],
         [df_sum['business'] if 'business' in df_sum.index else 0][0],
         [df_sum['entertainment'] if 'entertainment' in df_sum.index else 0][0],
         [df_sum['sport'] if 'sport' in df_sum.index else 0][0],
         [df_sum['tech'] if 'tech' in df_sum.index else 0][0],
         [df_sum['other'] if 'other' in df_sum.index else 0][0]]
    
    # Create plotly figure
    figure = {
        'data': [
            {'values': y,
             'labels': x, 
             'type': 'pie',
             'hole': .4,
             'name': '% of news articles',
             'marker': {'colors': ['rgb(62, 137, 195)',
                                   'rgb(167, 203, 232)',
                                   'rgb(197, 223, 242)',
                                   'rgb(51, 113, 159)',
                                   'rgb(64, 111, 146)',
                                   'rgb(31, 84, 132)']},

            }
        ],
        
        'layout': {
            'title': 'News articles by newspaper',
            'plot_bgcolor': colors['graph_background'],
            'paper_bgcolor': colors['graph_background'],
            'font': {
                    'color': colors['text'],
                    'size': '10'
            }
        }
        
    }
    
    return figure
    
    
@app.callback(
    [
    Output('textarea1a', 'href'),
    Output('textarea1a', 'children'),
    Output('textarea1b', 'children'),
    Output('textarea2a', 'href'),
    Output('textarea2a', 'children'),
    Output('textarea2b', 'children'),
    Output('textarea3a', 'href'),
    Output('textarea3a', 'children'),
    Output('textarea3b', 'children'),
    Output('textarea4a', 'href'),
    Output('textarea4a', 'children'),
    Output('textarea4b', 'children'),
    Output('textarea5a', 'href'),
    Output('textarea5a', 'children'),
    Output('textarea5b', 'children'),
    Output('textarea6a', 'href'),
    Output('textarea6a', 'children'),
    Output('textarea6b', 'children'),
    Output('textarea7a', 'href'),
    Output('textarea7a', 'children'),
    Output('textarea7b', 'children'),
    Output('textarea8a', 'href'),
    Output('textarea8a', 'children'),
    Output('textarea8b', 'children'),
    Output('textarea9a', 'href'),
    Output('textarea9a', 'children'),
    Output('textarea9b', 'children'),
    Output('textarea10a', 'href'),
    Output('textarea10a', 'children'),
    Output('textarea10b', 'children'),
    Output('textarea11a', 'href'),
    Output('textarea11a', 'children'),
    Output('textarea11b', 'children'),
    Output('textarea12a', 'href'),
    Output('textarea12a', 'children'),
    Output('textarea12b', 'children'),
    Output('textarea13a', 'href'),
    Output('textarea13a', 'children'),
    Output('textarea13b', 'children'),
    Output('textarea14a', 'href'),
    Output('textarea14a', 'children'),
    Output('textarea14b', 'children'),
    Output('textarea15a', 'href'),
    Output('textarea15a', 'children'),
    Output('textarea15b', 'children')
    ],
    [Input('intermediate-value', 'children')])
def update_textarea1(jsonified_df):
    
    df = pd.read_json(jsonified_df, orient='split')
    
    texts = []
    links = []
    preds_newsp = []
    
    for article in range(len(df)):
        texts.append(df.iloc[article]['Article Title'])
        links.append(df.iloc[article]['Article Link'])
        preds_newsp.append((df.iloc[article]['Prediction'].capitalize()) + ', ' + (df.iloc[article]['Newspaper']))

    while (len(texts) < 16):
        texts.append(None)
        links.append(None)
        preds_newsp.append(None)
    
    return \
        links[0], texts[0], preds_newsp[0],\
        links[1], texts[1], preds_newsp[1],\
        links[2], texts[2], preds_newsp[2],\
        links[3], texts[3], preds_newsp[3],\
        links[4], texts[4], preds_newsp[4],\
        links[5], texts[5], preds_newsp[5],\
        links[6], texts[6], preds_newsp[6],\
        links[7], texts[7], preds_newsp[7],\
        links[8], texts[8], preds_newsp[8],\
        links[9], texts[9], preds_newsp[9],\
        links[10], texts[10], preds_newsp[10],\
        links[11], texts[11], preds_newsp[11],\
        links[12], texts[12], preds_newsp[12],\
        links[13], texts[13], preds_newsp[13],\
        links[14], texts[14], preds_newsp[14]
           
    
    
# Loading CSS
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})
#####
# To here
#####

if __name__ == '__main__':
    app.run_server(debug=False)
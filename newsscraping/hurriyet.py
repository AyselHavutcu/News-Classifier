import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time

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
     {'Article Content': news_contents 
    })

# df_show_info
df_show_info = pd.DataFrame(
    {'Article Title': list_titles,
     'Article Link': list_links})


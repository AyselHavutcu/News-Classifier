# we r gonna fecth the news with beatifulsoup
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time


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


print(df_features)
print(df_show_info)

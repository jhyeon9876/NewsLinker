import requests
from bs4 import BeautifulSoup


def get_text_from_url(url: str):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    text = str(soup.find(class_='se-main-container').text).replace('\n\n', '')
    return text

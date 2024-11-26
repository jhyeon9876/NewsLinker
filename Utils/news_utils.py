from youtube_transcript_api import YouTubeTranscriptApi
from newspaper import Article
import requests
from bs4 import BeautifulSoup


def find_related_news(keyword: list, date: list):
    links = []

    # 뉴스 날짜 지정 아래는 1달 기준
    m_date = list(date)
    if m_date[1] == '1':
        m_date[0] = str(int(m_date[0]) - 1)
        m_date[1] = '12'
    else:
        m_date[1] = str(int(m_date[1]) - 1).zfill(2)

    url = f'https://search.naver.com/search.naver?where=news&query={"+".join(keyword)}&sm=tab_opt&sort=1&photo=0&field=0&pd=3&ds={".".join(m_date)}&de={".".join(date)}&docid=&related=0&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so%3Add%2Cp%3Aall&is_sug_officeid=0&office_category=0&service_area=0'
    ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'

    try:
        response = requests.get(url, headers={'User-Agent': ua})
        parser = BeautifulSoup(response.text, 'html.parser')
        tmp = parser.select('div > div > div.news_contents')
        for i in range(5):
            link = tmp[i].select_one('a.dsc_thumb').attrs['href']
            title = tmp[i].select_one('a.news_tit').attrs['title']
            img = ''
            try:
                img = tmp[i].select_one('a > img')['data-lazysrc']
            except Exception as e:
                print(e)
            finally:
                links.append({'link': link, 'title': title, 'img': img})
    except Exception as e:
        print(e)
    finally:
        return links


class News:
    def __init__(self, url):
        self.url = url

    def get_transcript(self):
        res = []
        try:
            video_id = self.url.split("=")[1]
            srt = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
            res = list(map(lambda x: x['text'], srt))
        except Exception as e:
            print(e)
        finally:
            return res

    def get_upload_date(self):
        res = requests.get(self.url)

        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            date = soup.find('meta', itemprop='datePublished')['content']
            # format: 2024-11-23T12:57:29-08:00
            return date.split('T')[0].split('-')
        else:
            return ['2024', '11', '23']

    @staticmethod
    def get_news_article(url: str) -> str:
        article = Article(url, language='ko')
        article.download()
        article.parse()
        return article.text.replace('\n', '')

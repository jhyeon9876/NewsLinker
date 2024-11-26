from flask import Flask, request
from Utils.news_utils import News, find_related_news
from Utils.ai_utils import Summary, KeywordExtractor, Similarity
import json

app = Flask(__name__)


# 뉴스링커 라우팅
@app.route('/newsLinker', methods=['GET'])
def do():
    try:
        print('=<Start: NewsLinker>==================')
        url: str = request.args.get('url')
        news = News(url)

        # 1. 자막 가져오기 + 업로드 날짜 가져오기
        transcript: list = news.get_transcript()
        date: list = news.get_upload_date()
        print(date)
        print('script: ', ' '.join(transcript))

        # 2. 자막 요약하기
        summarized_transcript: str = summary.summarize(' '.join(transcript))
        print('summarized_transcript: ', summarized_transcript)

        # 3. 자막에서 키워드 추출
        keywords: list = kw_extractor.extract_keywords(summarized_transcript)
        print('keywords: ', keywords)

        # 4. 검색
        related: list = find_related_news(keywords, date)
        print('news:', related)
        represent_article = news.get_news_article(related[0]['link'])

        # 5. 추출 기사 요약
        represent_summary: str = summary.summarize(represent_article)
        print('represent_summary: ', represent_summary)

        # 6. 결과 취합
        result = {
            'status': 'Success',
            'represent_article': {
                'title': related[0]['title'],
                'link': related[0]['link'],
                'summary': represent_summary,
                'img': related[0]['img'],
            },
            'news': [
                {
                    'title': i['title'],
                    'link': i['link'],
                    'img': i['img']
                } for i in related[1:]
            ]
        }
        print('=<End: NewsLinker>==================')
        # 리턴
        return json.dumps(result)
    except Exception as e:
        print(e)
        print('=<End: NewsLinker>==================')
        return json.dumps({'status': 'Fail', 'error': str(e)})


@app.route('/articleLinker', methods=['POST'])
def do2():
    try:
        print('=<Start: ArticleLinker>==================')
        article = request.json['text'] # 기사 전문
        date = request.json['date'].split('-') # 2024-11-11 이런 형태로 입력 예상
        more_details = request.json['more_details'] # T 혹은 F

        if more_details == 'T':
            #
            # 아직 손 안댐 여기 다시 건드려야 함
            #
            # kw: list = kw_extractor.extract_keywords(article)
            # print(f"전체키워드: {kw}")
            #
            # # 유사도 기반 문장 필터링
            # filtered_sentences = similarity.filtering(article, kw)
            # print(f"\n핵심문장 ({len(filtered_sentences)}):")
            # for sentence in filtered_sentences:
            #     print(f"- {sentence}")
            #
            # # 문장별 키워드 추출
            # sentence_keywords = similarity.extract_keywords_from_sentences(filtered_sentences, top_n=3)
            # print("\n문장별 키워드")
            # for sentence, keywords in sentence_keywords.items():
            #     print(f"{sentence}: {keywords}")
            pass
        elif more_details == 'F':
            res = list()
            topic_clusters = similarity.process_text_by_cluster(article, threshold=0.7, top_n=3)
            # 출력
            for topic, data in topic_clusters.items():
                related: list = find_related_news(data['keyword'], date)
                res.append([data['text'], related])

        result = {
            'status': 'Success',
            'related_news_by_cluster': res,
        }
        print('=<End: ArticleLinker>==================')
        return json.dumps(result)
    except Exception as e:
        print(e)
        print('=<End: ArticleLinker>==================')
        return json.dumps({'status': 'Fail', 'error': str(e)})


if __name__ == '__main__':
    print('Create Summary instance...')
    summary = Summary()
    print('Summary instance created')
    print('Create KeywordExtractor instance...')
    kw_extractor = KeywordExtractor()
    print('KeywordExtractor instance created')
    print('Create Similarity instance...')
    similarity = Similarity(kw_extractor)
    print('Similarity instance created')

    print('All necessary instances created!')

    app.run('0.0.0.0', port=5000)
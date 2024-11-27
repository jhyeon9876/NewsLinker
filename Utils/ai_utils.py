import nltk
from kiwipiepy import Kiwi
from keybert import KeyBERT
from transformers import BertModel, AutoModelForTokenClassification
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
kiwi = Kiwi()


class Summary:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('punkt_tab')
        self.model = AutoModelForSeq2SeqLM.from_pretrained('eenzeenee/t5-base-korean-summarization')
        self.tokenizer = AutoTokenizer.from_pretrained('eenzeenee/t5-base-korean-summarization')

    def summarize(self, article: str) -> str:
        array_input: list = ['summarize: ' + article]
        token = self.tokenizer(array_input, max_length=512, truncation=True, return_tensors='pt')
        output = self.model.generate(**token, num_beams=3, do_sample=True, min_length=10, max_length=64)
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        result: str = nltk.sent_tokenize(decoded_output.strip())[0]
        return result


class KeywordExtractor:
    def __init__(self):
        model = BertModel.from_pretrained('skt/kobert-base-v1')
        kw_extractor = KeyBERT(model)
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large-finetuned-conll03-english')
        ner_model = AutoModelForTokenClassification.from_pretrained('xlm-roberta-large-finetuned-conll03-english')
        ner_pipeline = pipeline('ner', model=ner_model, tokenizer=tokenizer)

        def extract_names(text) -> list:
            ner_results: list = ner_pipeline(text)
            names: list = [res['word'] for res in ner_results if res['entity'] == 'B-PER']
            return list(set(names))
        self.extract_names = extract_names

        def noun_and_name_extractor(text) -> str:
            results = []
            for t in text:
                result = kiwi.analyze(t)
                for token, pos, _, _ in result[0][0]:
                    if pos.startswith(('N', 'SL')) and len(token) > 1:
                        results.append(token)
            names_in_text: list = extract_names(" ".join(text))
            results.extend(names_in_text)
            return " ".join(results)
        self.noun_and_name_extractor = noun_and_name_extractor

        def keyword_one(text, top_n=5, use_mmr=True, diversity=0.5) -> list:
            keywords: list = kw_extractor.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 1),
                top_n=top_n,
                use_mmr=use_mmr,
                diversity=diversity
            )
            return keywords
        self.keyword_one = keyword_one

        self.kw_extractor = kw_extractor

    def extract_keywords(self, article, top_n=5, diversity=0.2, include_names=True) -> list:
        text: str = self.noun_and_name_extractor([article])
        keywords: list = self.keyword_one(text, top_n=top_n, use_mmr=True, diversity=diversity)
        if include_names:
            names_in_text = self.extract_names(" ".join([article]))
            keywords = [(name, 1.0) for name in names_in_text
                        if name not in [kw[0] for kw in keywords]] + keywords[:top_n]
        return [kw[0] for kw in keywords[:top_n]]


class Similarity:
    def __init__(self, extractor: KeywordExtractor, threshold=0.7):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.threshold: float = threshold
        self.extractor: KeywordExtractor = extractor

        def split_into_sentences(text) -> list:
            sentences = kiwi.split_into_sents(text)
            return [sent.text for sent in sentences]
        self.split_into_sentences = split_into_sentences

        def cluster_consecultive_sentences(sentences, threshold=0.7) -> list:
            embeddings = self.model.encode(sentences)
            cluster = []
            current_cluster = [0]
            for i in range(1, len(sentences)):
                sim = util.cos_sim(embeddings[i - 1], embeddings[i]).item()
                if sim >= threshold:
                    current_cluster.append(i)
                else:
                    cluster.append(current_cluster)
                    current_cluster = [i]
            if current_cluster:
                cluster.append(current_cluster)
            return cluster
        self.cluster_consecultive_sentences = cluster_consecultive_sentences

        def get_text_by_clusters(sentences, clusters) -> list:
            clustered_text = []
            for cluster in clusters:
                cluster_text = " ".join([sentences[i] for i in cluster])
                clustered_text.append(cluster_text)
            return clustered_text
        self.get_text_by_clusters = get_text_by_clusters

    def filtering(self, text: str, keywords: list) -> list:
        sentences = kiwi.split_into_sents(text)
        sentences = [sent.text for sent in sentences]

        keyword_embeddings = self.model.encode([kw[0] for kw in keywords])
        sentence_embeddings = self.model.encode(sentences)

        filtered_sentences = []
        for i, sentence in enumerate(sentences):
            # 유사도 계산
            similarities = util.cos_sim(sentence_embeddings[i], keyword_embeddings).max().item()
            # 조건: 키워드 포함 여부 또는 유사도 임계값 이상
            if any(kw[0] in sentence for kw in keywords) or similarities >= self.threshold:
                filtered_sentences.append(sentence)

        return filtered_sentences

    # 문장별 키워드 추출 함수
    def extract_keywords_from_sentences(self, sentences, top_n=3) -> dict:
        sentence_keywords = {}
        for idx, sentence in enumerate(sentences):
            filtered_sentence = self.extractor.noun_and_name_extractor([sentence])
            keywords = self.extractor.kw_extractor.extract_keywords(
                filtered_sentence, keyphrase_ngram_range=(1, 1), top_n=top_n, use_mmr=True, diversity=0.2
            )
            sentence_keywords[f"문장 {idx + 1}"] = [kw[0] for kw in keywords]
        return sentence_keywords

    def process_text_by_cluster(self, text, threshold=0.7, top_n=3) -> dict:
        sentences = self.split_into_sentences(text)
        clusters = self.cluster_consecultive_sentences(sentences, threshold)
        clustered_texts = self.get_text_by_clusters(sentences, clusters)

        topic_keywords = {}
        for idx, cluster_text in enumerate(clustered_texts):
            keywords = self.extractor.extract_keywords(cluster_text, top_n=top_n)
            topic_keywords[f"주제 {idx + 1}"] = {
                "text": cluster_text,
                "keyword": keywords
            }
        return topic_keywords


from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

INDEX_NAME = "mujung"
TEXT_DATA_PATH = "mujung.txt"

es = Elasticsearch("http://localhost:9200", request_timeout=20)

# 1) 인덱스 재생성
if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)

index_body = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "filter": {
                "korean_pos_filter": {
                    "type": "nori_part_of_speech",
                    # 책에서는 '조사/어미 제거'를 명확히 보여주기 위해 stoptags를 명시 권장
                    "stoptags": [
                        "J", "E", "IC", "MAJ", "MM",
                        "SP", "SSC", "SSO", "SC", "SE",
                        "XPN", "XSA", "XSN", "XSV"
                    ]
                }
            },
            "analyzer": {
                "ko_analyzer": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "filter": ["korean_pos_filter"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "line": {"type": "text", "analyzer": "ko_analyzer"},
            "line_number": {"type": "integer"}
        }
    }
}

es.indices.create(index=INDEX_NAME, body=index_body)

# 2) bulk 인덱싱 (메모리 절약: generator)
def gen_actions():
    with open(TEXT_DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield {
                "_index": INDEX_NAME,
                "_id": i,
                "_source": {
                    "line": line,
                    "line_number": i
                }
            }

bulk(es, tqdm(gen_actions(), desc="indexing"))

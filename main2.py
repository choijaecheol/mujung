from tqdm import tqdm
from elasticsearch_dsl import Document, Text, Integer, connections, analyzer, Index, DenseVector, Search
from elasticsearch.helpers import bulk
from openai import OpenAI
import os

# =========================
# 설정
# =========================
INDEX_NAME = "mujung"
TEXT_DATA_PATH = "mujung.txt"  # 한국어 텍스트 파일 경로
EMBED_MODEL = "text-embedding-3-small"   # 1536 dims

# OpenAI 클라이언트 (환경변수 OPENAI_API_KEY 필요)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Elasticsearch 연결
es = connections.create_connection(hosts=["http://localhost:9200"], timeout=20)

# =========================
# 임베딩 함수
# =========================
def get_embedding(text: str) -> list[float]:
    resp = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding

# =========================
# 0. nori 어날라이저 정의
# =========================
ko_analyzer = analyzer(
    "ko_analyzer",
    tokenizer="nori_tokenizer",
    filter=["nori_part_of_speech"],
)

# =========================
# 4.1 인덱스 생성 
# =========================
class MujungLineWithEmbedding(Document):
    line = Text(analyzer=ko_analyzer)
    line_embedding = DenseVector(dims=1536)
    line_number = Integer()

    class Index:
        name = INDEX_NAME

# (인덱스 재생성)
if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)
    print(f"Index {INDEX_NAME} deleted.")

index = Index(INDEX_NAME)
index.settings(
    number_of_shards=1,
    number_of_replicas=0,
    analysis={
        "analyzer": {
            "ko_analyzer": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "filter": ["nori_part_of_speech"],
            }
        }
    },
)

index.document(MujungLineWithEmbedding)
index.create()

# =========================
# 4.2 인덱스 등록 
# =========================
docs = []

with open(TEXT_DATA_PATH, "r", encoding="utf-8") as f:
    for i, line in tqdm(enumerate(f)):
        line = line.strip()
        if len(line) > 0:
            emb = get_embedding(line)
            doc = MujungLineWithEmbedding(
                meta={'id': i}, line=line, line_number=i, line_embedding=emb
            )
            docs.append(doc.to_dict(include_meta=True))

bulk(es, docs, index=INDEX_NAME)

# =========================
# 4.3 검색 쿼리 실행 
# =========================
client = connections.create_connection(hosts=["http://localhost:9200"], timeout=20)
query_text = "주인공이 슬픔을 느끼는 장면"
query_vector = get_embedding(query_text)

S = Search(using=client, index="mujung").query(
    "script_score",
    query={"match_all": {}},
    script={
        "source": "cosineSimilarity(params.query_vector, 'line_embedding')",
        "params": {"query_vector": query_vector}
    }
)
response = S.execute()

for hit in response:
    print(f"{hit.meta.score:.2f}: {hit.line}")
import os
from typing import List, Dict, Any
import openai
from elasticsearch import Elasticsearch
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)


openai.api_key = os.getenv('OPENAI_API_KEY')

INDEX_NAME = "mujung"
es = Elasticsearch("http://localhost:9200")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

def create_hybrid_query(
    query, embedding_model,
    vector_weight=0.7, keyword_weight=0.3, size=5
):
    query_vec = embedding_model.embed_query(query)
    return {
        "bool": {
            "should": [
                {
                    "knn": {
                        "field": "line_embedding",
                        "query_vector": query_vec,
                        "k": size,
                        "num_candidates": 100,
                        "boost": vector_weight
                    }
                },
                {
                    "match": {
                        "line": {
                            "query": query,
                            "boost": keyword_weight
                        }
                    }
                }
            ]
        }
    }

def hybrid_search(query, k=5, vector_weight=0.7, keyword_weight=0.3):
    hybrid_query = create_hybrid_query(
        query, embedding_model, vector_weight, keyword_weight, size=k
    )
    body = {'query': hybrid_query, 'size': k}
    results = es.search(index=INDEX_NAME, body=body)
    return [
        {
            "text": hit["_source"]["line"],
            "line_number": hit["_source"]["line_number"],
            "score": hit["_score"],
        }
        for hit in results["hits"]["hits"]
    ]

def answer_with_rag_chain(question, k=5):
    # 검색
    search_results = hybrid_search(query=question, k=k,
                                   vector_weight=0.7, keyword_weight=0.3)
    search_results.sort(key=lambda x: x['line_number'])
    context = "\n".join([r['text'] for r in search_results])

    # 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 한국 문학 '무정'의 전문가입니다."),
        ("user", """텍스트 단편을 참고하여 질문에 답변해 주세요.

텍스트 단편:
---
{context}
---

질문: {question}""")
    ])

    # LCEL (LangChain Expression Language) 방식으로 체인 구성
    chain = prompt | llm
    response = chain.invoke({'context': context, 'question': question})
    return response


# 실행 예
if __name__ == "__main__":
    query = "주인공은 어떤 감정을 느끼고 있었나요?"
    print("=== 하이브리드 검색 테스트 ===")
    results = hybrid_search(query, k=5)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['text']} (score: {result['score']:.4f})")

    print("\n=== RAG를 이용한 답변 ===")
    answer = answer_with_rag_chain(query)
    print(f"답변: {answer}")

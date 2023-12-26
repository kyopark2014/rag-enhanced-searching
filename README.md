# 한영 동시 검색 및 인터넷 검색을 활용하여 RAG 성능 향상시키기

기업의 중요한 문서를 검색하여 편리하게 활용하기 위하여 LLM(Large Language Model)을 활용하는 기업들이 늘어나고 있습니다. 기업의 모든 데이터를 사전학습하는것은 비용 및 시간에 대한 제약뿐 아니라 데이터 보안 면에서도 바람직하지 않습니다. 따라서, [RAG(Retrieval-Augmented Generation)](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html)의 지식저장소(Knowledge Store)를 이용하여 다수의 문서를 안전하게 검색하여 관련된 문서(Relevant docuents)를 추출한 후에 LLM으로 용도에 맞게 활용합니다.

RAG의 지식저장소에는 한국어로된 문서뿐 아니라 영어등 다국어 문서가 등록될수 있는데, 질문이 한국어라면 한국어 문서만 검색이 가능하고, 영어 문서는 별도로 검색하여야 합니다. 또한, RAG의 지식저장소에 없는 내용을 문의한 경우에 인터넷으로 쉽게 검색하여 답할수 있는 내용을 답하지 못하는 경우도 있습니다. 여기에서는 한영 동시 검색을 이용하여 한국어로 질문시에 영어 문서까지 검색하여 통합하여 결과를 보여주는 방법과 인터넷 검색을 통하여 RAG에 없는 데이터를 실시간으로 조회하여 결과를 보여주는 방법을 설명합니다. 이를 통해, RAG의 검색 성능을 향상시키고 LLM을 통한 Question/Answering을 진행할때에 사용자 편의성을 높일 수 있습니다. 

한영 동시 검색의 최종 결과가 한국어야 합니다. 따라서, 한국어와 영어로 검색한 결과의 문서를 하나의 한국어 결과를 얻기 위해서는 영어로 검색한 결과를 한국어로 번역하여 LLM에 context로 제공하여야 합니다. 여기에서는 영어를 한국어로 번역하기 위하여 LLM의 propmt를 활용합니다. 따라서 영어로 N개의 관련된 문서(Relevant Document)가 검색되면 N번의 request를 통해 LLM으로 번역하는 과정이 필요합니다. 이를 연속적(sequencial)하게 처리하면 사용자의 질문과 답변사이의 지연시간이 길어져서 사용성이 나빠집니다. 따라서, 여기에서는 Multi-Region LLM을 활용하여 다수의 LLM에 번역을 요청하여 시간을 단축합니다.


## Architecture 개요

한국어와 영어로된 문서들이 있을 경우에 한국어 문서는 한국어로 질문을 하고 영어 문서는 영어로 검색해야 원하는 결과를 얻을 수 있습니다. 한국어를 사용하는 사용자의 편의를 위해 아래 2가지 사항의 고려가 필요합니다. 

1) 영어로 질문시 한국어로 번역한 결과를 같이 보여줘서 이해를 돕습니다.

2) 한국어로 질문시, 한국어 문서를 먼저 검색하고 영어로 검색한 결과를 함께 반영한 결과를 얻으므로 사용성을 향상시킵니다.

이를 위해서는 영어결과를 한국어로 번역할 수 있어야 하며, 한국어 질문을 영어로 번역해서 다시 질의하여 얻어진 결과를 하나로 정리해서 제공할 수 있어야 합니다. 결과적으로 관련 문서의 증가 및 추가적인 번역 작업으로 전체적인 지연 시간이 증가할 수 있습니다. 따라서, LLM을 이용한 한영 번역한 것을 context로 활용할 수 있어야 하고, 한국어와 영어로 각각 검색한 결과를 한국어로 통합하여야 하고, 늘어난 지연시간을 단축할 수 있어야 합니다. 

RAG는 지식저장소에서 관련된문서들(Relevant documents)를 추출하여 LLM의 context로 활용합니다. 따라서, context에 없는 내용을 질문하게 되면 모른다고 답변하여야 합니다. 이는 매우 정상적은 RAG의 활용 방식이지만, 인터넷으로 쉽게 검색되는 내용을 모른다고 하는것은 사용성에서 좋지 않을 수 있습니다. 이와같이 RAG에 관련된 문서가 없는 경우에 인터넷 검색을 통해 얻은 내용을 마치 RAG처럼 관련된 문서로 제공하면 사용성을 개선할 수 있습니다. 이를 위해서는 인터넷 검색 API를 연동하고 이를 RAG와 결합하여 결과를 보여줄 수 있어야 합니다.

아래는 한영 동시 검색 및 인터넷 검색을 활용할 수 있는 Architecture를 보여주고 있습니다. 

<img src="https://github.com/kyopark2014/rag-enhanced-searching/assets/52392004/f7590d5d-b37d-492b-9bc3-1d4a1b76fff7" width="800">

사용자의 요청은 Amazon API Gateway를 통해 전달되고, AWS Lambda를 통해 질문을 통해 처리됩니다. 한영 동시 검색을 위해 다수의 문서를 번역하여야 하므로 지연시간을 단축시키기 위하여 Multi-Region LLM을 활용합니다. 여기서는 us-east-1, us-west-2, ap-northeast-1, us-central-1의 Bedrock을 활용합니다. RAG의 지식저장소로는 Amazon OpenSearch와 Amazon Kendra를 활용합니다. OpenSearch는 매우 빠르고 좋은 성능의 검색 능력을 제공할 수 있고, Kendra는 다양한 데이터 소스를 활용하여 많은 데이터를 쉽게 모으고 관리할 수 있습니다. Multi-RAG를 조회시간을 단축하기 위하여 다중 Thread를 활용하여 동시에 조회를 수행합니다. 사용자와 Chatbot의 대화이력은 DynamoDB에 저장되고, 원할한 대화를 위해 활용됩니다. 또한, 여러개의 관련된 문서가 있으면, 문서의 우선순위를 정하여서 관련도가 높은 문서를 선택하여야 하여 상단에 놓을수 있어야 합니다. Faiss의 Similarity Search를 활용하여 Reranking 하도록 우선성 검색(Priority Search)을 하면 소수의 문서에 대한 Embedding이 필요하지만, 정략적으로 사용할 수 있는 점수(score)로 관련성 있는 문서를 선택 및 정렬할 수 있습니다. 또한 Lambda의 process와 memory를 활용하므로 별도로 비용이 추가되지 않습니다. 

이때의 상세한 Signal Flow는 아래와 같습니다.

1. 사용자의 질문(question)은 API Gateway를 통해 Lambda에 https POST 방식으로 전달됩니다. Lambda는 JSON body에서 질문을 읽어옵니다. 이때 사용자의 이전 대화이력이 필요하므로 Amazon DynamoDB에서 읽어옵니다. DynamoDB에서 대화이력을 로딩하는 작업은 처음 1회만 수행합니다.
2. 사용자의 대화이력을 반영하여 사용자와 Chatbot이 interactive한 대화를 할 수 있도록, 대화이력과 사용자의 질문으로 새로운 질문(Revised Question)을 생성합니다. 이때 LLM에 대화이력(chat history)를 LLM에 context로 제공하고 적절한 Prompt를 이용하여 새로운 질문을 생성합니다.
3. 새로운 질문(Revised question)으로 OpenSearch와 Kendra에 질문을 하여 관련된 문서(Relevant Documents)를 얻습니다. 이때 시간 단축을 위하여 멀티 Thread를 이용하여 동시에 질문하여 지연시간을 단축합니다. 
4. 질문이 한국어인 경우에 Revised question을 영어로 번역합니다.
5. 번역된 새로운 질문(translated revised question)을 이용하여 다시 OpenSearch와 Kendra에 다시 질문을 전닳합니다.
6. 번역된 질문으로 얻은 관련된 문서가 영어 문서 일 경우에, LLM을 통해 번역을 수행합니다. 관련된 문서가 여러개이므로 Multi-Region의 LLM들을 활용하여 지연시간을 최소화 합니다.
7. 한국어로 질문으로 얻은 N개의 관련된 문서와, 영어로 된 N개의 관련된 문서의 합은 최대 2xN개입니다. 이 문서를 가지고 context window 크기에 맞도록 문서를 선택합니다. 이때 관련되가 높은 문서가 context의 상단에 가도록 배치합니다.
8. 관련도가 일정이하인 문서는 버리므로, 한개의 RAG의 문서도 선택되지 않을 수 있습니다. 이때에는 Google Seach API를 통해 인터넷 검색을 수행하고 하고, 이때 얻어진 문서들을 RAG처럼 Priority Search를 하여 선택한 후에 RAG 처럼 활용할 수 있습니다.
9. 선택된 관련된 문서들(Selected relevant documents)로 Context를 생성한 후에 새로운 질문(Revised question)과 함께 LLM에 전달하여 사용자의 질문에 대한 답변을 생성하여 사용자에게 전달합니다.

<img src="https://github.com/kyopark2014/rag-enhanced-searching/assets/52392004/fb2d4d52-afb6-4ac3-ab7d-904b5d348469" width="900">




## 한영 Dual Search

revised question을 먼저 영어로 변환하여 Mult-RAG를 통해 조회합니다. 영어와 한글 문서를 모두 가지고 있는 Knowlege Store는 한국어 문서도 관련 문서로 제공할 수 있으므로, 영어로된 관련된 문서(Relevant Document)를 찾아서 한국어로 변환합니다. 이후, 한국어 검색으로 얻어진 결과에 추가합니다. 이렇게 되면 한국어로 검색했을때보다 2배의 관련된 문서들을 가지게 됩니다. 

```python
translated_revised_question = traslation_to_english(llm=llm, msg=revised_question)

relevant_docs_using_translated_question = get_relevant_documents_using_parallel_processing(question=translated_revised_question, top_k=top_k)

relevant_docs_using_translated_question = []
for reg in capabilities:            
    if reg == 'kendra':
        rel_docs = retrieve_from_kendra(query=translated_revised_question, top_k=top_k)      
        print('rel_docs (kendra): '+json.dumps(rel_docs))
    else:
        rel_docs = retrieve_from_vectorstore(query=translated_revised_question, top_k=top_k, rag_type=reg)
        print(f'rel_docs ({reg}): '+json.dumps(rel_docs))

    if(len(rel_docs)>=1):
        for doc in rel_docs:
            relevant_docs_using_translated_question.append(doc)    

if len(relevant_docs_using_translated_question)>=1:
    for i, doc in enumerate(relevant_docs_using_translated_question):
        if isKorean(doc)==False:
            translated_excerpt = traslation_to_korean(llm=llm, msg=doc['metadata']['excerpt'])
            doc['metadata']['translated_excerpt'] = translated_excerpt
            relevant_docs.append(doc)
        else:
            print(f"original {i}: {doc}")
            relevant_docs.append(doc)
```

그런데, 영어로 번역된 질문으로 조회한 Relevant Document의 숫자만큰 한국어로 번역이 필요하므로 프로세싱 시간이 관련된 문서수만큼 증가하는 이슈가 발생합니다. 이는 사용성을 저하 시키므로 개선이 필요합니다. 여기에서는 Multi-Region LLM을 활용하여 4개의 리전의 LLM을 활용하여 RAG 문서를 한국어로 번역하는 시간을 줄입니다. 아래와 같이 영어로 질문을 한 후에 얻어진 문서들에서 번역이 필요한 리스트를 추출합니다. 이후 multi thread를 이용하여 각 리전으로 LLM에 번역을 요청합니다. 

```python
docs_translation_required = []
if len(relevant_docs_using_translated_question) >= 1:
    for i, doc in enumerate(relevant_docs_using_translated_question):
        if isKorean(doc) == False:
            docs_translation_required.append(doc)
        else:
            relevant_docs.append(doc)
translated_docs = translate_relevant_documents_using_parallel_processing(docs_translation_required)

for i, doc in enumerate(translated_docs):
  relevant_docs.append(doc)

def translate_relevant_documents_using_parallel_processing(docs):
    selected_LLM = 0
    relevant_docs = []    
    processes = []
    parent_connections = []
    for doc in docs:
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        llm = get_llm(profile_of_LLMs, selected_LLM)
        process = Process(target=translate_process_from_relevent_doc, args=(child_conn, llm, doc))            
        processes.append(process)

        selected_LLM = selected_LLM + 1
        if selected_LLM == len(profile_of_LLMs):
            selected_LLM = 0

    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        doc = parent_conn.recv()
        relevant_docs.append(doc)    

    for process in processes:
        process.join()
    
    return relevant_docs
```

결과적으로 4배의 속도 향상이 있었습니다. (추후 결과를 수치로 제시할것) 

관련된 문장의 숫자가 늘어났으므로 context로 활용할 문서를 추출합니다. 아래의 priority_search()는 Faiss의 similarity search를 이용하여 관련문서를 reranking하는 동작을 수행합니다. 

```python
selected_relevant_docs = []
if len(relevant_docs)>=1:
    selected_relevant_docs = priority_search(revised_question, relevant_docs, bedrock_embeddings)

def priority_search(query, relevant_docs, bedrock_embeddings):
    excerpts = []
    for i, doc in enumerate(relevant_docs):
        if doc['metadata']['translated_excerpt']:
            content = doc['metadata']['translated_excerpt']
        else:
            content = doc['metadata']['excerpt']
        
        excerpts.append(
            Document(
                page_content=content,
                metadata={
                    'name': doc['metadata']['title'],
                    'order':i,
                }
            )
        )  

    embeddings = bedrock_embeddings
    vectorstore_confidence = FAISS.from_documents(
        excerpts,  # documents
        embeddings  # embeddings
    )            
    rel_documents = vectorstore_confidence.similarity_search_with_score(
        query=query,
        k=top_k
    )

    docs = []
    for i, document in enumerate(rel_documents):

        order = document[0].metadata['order']
        name = document[0].metadata['name']
        assessed_score = document[1]

        relevant_docs[order]['assessed_score'] = int(assessed_score)

        if assessed_score < 200:
            docs.append(relevant_docs[order])    

    return docs
```            

선택된 관련문서를 이용해 Prompt를 생성한 후에 LLM에 질의하여 영한 검색을 통한 결과를 얻을 수 있습니다.

```python
relevant_context = ""
for document in selected_relevant_docs:
    if document['metadata']['translated_excerpt']:
        content = document['metadata']['translated_excerpt']
    else:
    content = document['metadata']['excerpt']

relevant_context = relevant_context + content + "\n\n"
print('relevant_context: ', relevant_context)

stream = llm(PROMPT.format(context = relevant_context, question = revised_question))
msg = readStreamMsg(connectionId, requestId, stream)            
```



### 영어로 얻어진 문장을 한국어로 번역

```text
pip install google-api-python-client
```

```python
def traslation_to_english(llm, msg):
    PROMPT = """\n\nHuman: 다음의 <article>를 English로 번역하세요. 머리말은 건너뛰고 본론으로 바로 들어가주세요. 또한 결과는 <result> tag를 붙여주세요.

    <article>
    {input}
    </article>
                        
    Assistant:"""

    try: 
        translated_msg = llm(PROMPT.format(input=msg))
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to translate the message")
    
    return translated_msg[translated_msg.find('<result>')+9:len(translated_msg)-10]
```

### Google Search API를 이용한 검색기능

Multi-RAG로 검색하여 Relevant Document가 없는 경우에 Google API를 이용해 검색한 결과를 RAG에서 사용합니다. 상세한 내용은 [Google Search API](./GoogleSearchAPI.md)에서 확인합니다. 여기서, assessed_score는 priority search시 FAISS의 Score로 업데이트 됩니다.

[api_key](https://developers.google.com/custom-search/docs/paid_element?hl=ko#api_key)에서 [키 가져오기] - [Select or create project]를 선택하여 Google API Key를 가져옵니다. 만약 기존 키가 없다면 새로 생성합니다.

[새 검색엔진 만들기](https://programmablesearchengine.google.com/controlpanel/create?hl=ko)에서 검색엔진을 설정합니다. 이때, 검색할 내용은 "전체 웹 검색"을 선택하여야 합니다.


```python
from googleapiclient.discovery import build

google_api_key = os.environ.get('google_api_key')
google_cse_id = os.environ.get('google_cse_id')

api_key = google_api_key
cse_id = google_cse_id

relevant_docs = []
try:
    service = build("customsearch", "v1", developerKey = api_key)
    result = service.cse().list(q = revised_question, cx = cse_id).execute()
    print('google search result: ', result)

    if "items" in result:
        for item in result['items']:
            api_type = "google api"
            excerpt = item['snippet']
            uri = item['link']
            title = item['title']
            confidence = ""
            assessed_score = ""

            doc_info = {
                "rag_type": 'search',
                "api_type": api_type,
                "confidence": confidence,
                "metadata": {
                    "source": uri,
                    "title": title,
                    "excerpt": excerpt,                                
                },
                "assessed_score": assessed_score,
            }
        relevant_docs.append(doc_info)
```


### 영어로 질문시 한글 결과를 같이 보여주기

영어로 질의시 영어 문서들을 조회할 수 있습니다. 결과가 한국어/영어인것을 확인한 후에 한국어가 아니라면 LLM에 문의하여 아래와 같이 한국어로 본역한 후에 결과에 추가하여 같이 보여줍니다.

```python
if isKorean(msg)==False:
  translated_msg = traslation_to_korean(llm, msg)

msg = msg+'\n[한국어]\n'+translated_msg

def isKorean(text):
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))

    if word_kor and word_kor != 'None':
        return True
    else:
        return False

def traslation_to_korean(llm, msg):
    PROMPT = """\n\nHuman: Here is an article, contained in <article> tags. Translate the article to Korean. Put it in <result> tags.
            
    <article>
    {input}
    </article>
                        
    Assistant:"""

    try: 
        translated_msg = llm(PROMPT.format(input=msg))
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to translate the message")
    
    msg = translated_msg[translated_msg.find('<result>')+9:len(translated_msg)-10]
    
    return msg.replace("\n"," ")
```


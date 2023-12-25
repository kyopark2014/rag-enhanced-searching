# 한영 동시 검색 및 인터넷 검색을 활용하여 RAG 성능 향상시키기

LLM에 질문시 RAG를 통해 사전 학습되지 않은 정보들을 검색할 수 있습니다. 만약 한글로 검색한다면 RAG의 Knowledge Store에 있는 한글 문서들을 검색하게 됩니다. 따라서 다수의 영어를 포한한 다른 언어를 동시에 검색하는 것이 필요합니다.

RAG의 Knowledge Store에 없는 질문을 한다면 LLM은 모른다고 답변하게 됩니다. 하지만, 이 경우에 인터넷 검색을 통해 Knowledge Stroe의 부족한 면을 채울수 있다면 검색 성능을 향상 시킬수 있습니다.

## 한영 동시 검색

한글로 질문시 영어문서는 번역해서 조회후 영어답변을 다시 한글로 하면됨

한글로 질문시 영어/한글을 검색하려면 영어/한글 각각 검색해서 얻은 문서를 한글로 번역후에 답변을 구해야함.

관련문서가 적으면 결과에 영향을 주고, 번역을 많이 하려면 속도가 느려짐

이때 multi region이나 pf을 써야함

multi region을 round robin으로 쓰는 방식이 아니라 multi thread로 해야함

region별로 bedrock client를 만들어서, thread를 리전수만큼 해야함


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

## 한영 Dual Search

revised question을 먼저 영어로 변환하여 Mult-RAG를 통해 조회합니다. 영어와 한글 문서를 모두 가지고 있는 Knowlege Store는 한국어 문서도 관련 문서로 제공할 수 있으므로, 영어로된 관련된 문서(Relevant Document)를 찾아서 한국어로 변환합니다. 이후, 한국어 검색으로 얻어진 결과에 추가합니다. 이렇게 되면 한국어로 검색했을때보다 2배의 관련된 문서들을 가지게 됩니다. 

```python
translated_revised_question = traslation_to_english(llm=llm, msg=revised_question)

relevant_docs_using_translated_question = get_relevant_documents_using_parallel_processing(question=translated_revised_question, top_k=top_k)

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

이제 관련문서들중에 사용할 문서들을 추출합니다.

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

### 영어로 얻어진 문장을 한국어로 번역

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

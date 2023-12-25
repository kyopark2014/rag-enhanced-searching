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

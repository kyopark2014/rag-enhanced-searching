import json
import boto3
import os
import traceback
from botocore.config import Config
from urllib.parse import unquote_plus

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
meta_prefix = "metadata/"
kendra_region = os.environ.get('kendra_region', 'us-west-2')

opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
opensearch_url = os.environ.get('opensearch_url')
kendraIndex = os.environ.get('kendraIndex')

from opensearchpy import OpenSearch
def delete_index_if_exist(index_name):
    client = OpenSearch(
        hosts = [{
            'host': opensearch_url.replace("https://", ""), 
            'port': 443
        }],
        http_compress = True,
        http_auth=(opensearch_account, opensearch_passwd),
        use_ssl = True,
        verify_certs = True,
        ssl_assert_hostname = False,
        ssl_show_warn = False,
    )

    if client.indices.exists(index_name):
        print('delete opensearch document index: ', index_name)
        response = client.indices.delete(
            index=index_name
        )
        print('removed index: ', response)    
    else:
        print('no index: ', index_name)

# Kendra
kendra_client = boto3.client(
    service_name='kendra', 
    region_name=kendra_region,
    config = Config(
        retries=dict(
            max_attempts=10
        )
    )
)

# load csv documents from s3
def lambda_handler(event, context):
    print('event: ', event)

    documentIds = []
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        # translate utf8
        key = unquote_plus(record['s3']['object']['key']) # url decoding
        print('bucket: ', bucket)
        print('key: ', key)

        # get metadata from s3
        metadata_key = meta_prefix+key+'.metadata.json'
        print('metadata_key: ', metadata_key)

        metadata_obj = s3.get_object(Bucket=bucket, Key=metadata_key)
        metadata_body = metadata_obj['Body'].read().decode('utf-8')
        metadata = json.loads(metadata_body)
        print('metadata: ', metadata)
        documentId = metadata['DocumentId']
        print('documentId: ', documentId)
        documentIds.append(documentId)

        # delete metadata
        print('delete metadata: ', metadata_key)
        try: 
            result = s3.delete_object(Bucket=bucket, Key=metadata_key)
            # print('result of metadata deletion: ', result)
        except Exception:
            err_msg = traceback.format_exc()
            print('err_msg: ', err_msg)
            raise Exception ("Not able to delete documents in Kendra")
  
        # delete document index of opensearch
        index_name = "rag-index-"+documentId
        # print('index_name: ', index_name)
        delete_index_if_exist(index_name)

    # delete kendra documents
    print('delete kendra documents: ', documentIds)
    try: 
        result = kendra_client.batch_delete_document(
            IndexId = kendraIndex,
            DocumentIdList=[
                documentId,
            ],
            #DataSourceSyncJobMetricTarget={
            #    'DataSourceId': '850d68bd-464e-4831-bc4a-ccc8c59d8fe1'
            #}
        )
        print('result: ', result)
    except Exception:
        err_msg = traceback.format_exc()
        print('err_msg: ', err_msg)
        raise Exception ("Not able to delete documents in Kendra")
    
    return {
        'statusCode': 200
    }

import json
import sklearn
import boto3
import os
import json
import pickle
s3 = boto3.client('s3')
s3_bucket = os.environ['s3_bucket']
model_name = os.environ['model_name']
temp_file_path = '/tmp/' + model_name
from sklearn.ensemble import RandomForestClassifier
def lambda_handler(event, context):
    # Parse input
    body = event['body']
    assist , asx , asy , x , y = int(json.loads(body)['assist']), float(json.loads(body)['asx']), float(json.loads(body)['asy']) , float(json.loads(body)['x']), float(json.loads(body)['y'])
    fin, hed, inv, nom = int(json.loads(body)['fin']), int(json.loads(body)['hed']), int(json.loads(body)['inv']) ,int(json.loads(body)['nom']) 
    s3.download_file(s3_bucket, model_name, temp_file_path)
    with open(temp_file_path, 'rb') as f:
        model = pickle.load(f)
    # Predict class
    prediction = model.predict_proba([[assist,asx,asy,fin,hed,inv,nom,x,y]])[0][1]
    return {
        "statusCode": 200,
        "body": json.dumps({
            "prediction": prediction,
        }),
    }
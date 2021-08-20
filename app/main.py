import pickle
from datetime import datetime

# import boto3
import numpy as np
from fastapi import FastAPI
from numpy import NaN as NA
from joblib import load

app = FastAPI()

S3_BUCKET = 'buzzvil-ml'
commerce_name = 'coupang-partners'
# s3_client = boto3.client('s3')

# s3_client.download_file(S3_BUCKET, 'Targeting/rsc/model/feature_stat.pkl', 'feature_stat.pkl')
with open('feature_stat.pkl', 'rb') as handle:
    feature_stat = pickle.load(handle)

# data processor
# s3_client.download_file(S3_BUCKET, 'Targeting/rsc/model/one-hot-lr-plus.joblib', 'one-hot.joblib')
# s3_client.download_file(S3_BUCKET, 'Targeting/rsc/model/scaler-lr-plus.joblib', 'scaler.joblib')
# s3_client.download_file(S3_BUCKET, 'Targeting/rsc/model/ip-hasher-lr-plus.joblib', 'ip-hasher.joblib')
scaler = load('scaler.joblib')
hasher = load('ip-hasher.joblib')
encoder = load('one-hot.joblib')

model_dir = f'{commerce_name}-ctr-clf.pkl'
# s3_client.download_file(S3_BUCKET, f'Targeting/rsc/model/{commerce_name}-lr-plus.pkl', model_dir)

with open(model_dir, 'rb') as f:
    lr = pickle.load(f)

def get_request():
    return {'viewer_avg_ctr': [0.396777, NA] * 50, # float
            'viewer_clk_count': [2979, NA] * 50, # int
            'viewer_imp_count': [7508, NA] * 50, # int
            'sex': ['F', 'Null'] * 50, # string
            'year_of_birth': [1965, NA] * 50, # int
            'ip': [3546977248, 'Null'] * 50, # string
            'registered_days': [NA, NA] * 50, # int
            'network_type': ['wifi', 'Null'] * 50, # string
            'carrier': ['kt', 'verizon'] * 50, # string
            'region': ['경기도 평택시', 'Null'] * 50, # string
            'device_name': ['SM-G950N', 'Null'] * 50, # string
            'creative_name': ['AirPods Pro 할인', 'AirPods Pro 할인'] * 50, # string
            'lineitem_id': ['1811555', '1811556'] * 50, # string
            'ad_group_id': ['18827', '18827'] * 50, # string
            'lineitem_category': ['commerce', 'finance'] * 50, # string
            'lineitem_avg_ctr': [0.0163038, NA] * 50, # float
            'lineitem_clk_count': [635, NA] * 50, # int
            'lineitem_imp_count': [38948, NA] * 50, # int
            'unit_id': ['16068189836608', '16068189836608'] * 50, # string
            'app_id': ['471890553154507', '471890553154507'] * 50, # string
            'organization_id': ['6111', '1'] * 50, # string
            'impressed_at_hour': [0, 3] * 50, # int
            'impressed_at_weekday': [1, 5] * 50} # int

def getinferencescores():
    request = get_request()
    # Fill Nulls to 'Unknown'
    for feature in ['sex', 'lineitem_category', 'network_type', 'region', 'carrier', 'ad_group_id',
                    'lineitem_id', 'ip', 'device_name', 'creative_name']:
        array = np.array(request[feature], dtype='<U16')
        array[array == 'Null'] = 'Unknown'
        request[feature] = array

    # Preprocess Registered Days
    array = np.array(request['registered_days'])
    array[np.isnan(array)] = feature_stat['avg_reg_days']
    request['registered_days'] = array

    # Preprocess YoB to Age
    array = np.array(request['year_of_birth'])
    array[np.isnan(array)] = feature_stat['avg_yob']
    request['age'] = datetime.now().year - array
    np.clip(array, 1, 99, out=array)
    del request['year_of_birth']

    # Preprocess carrier, region, device_name
    array = np.array(request['carrier'], dtype='<U16')
    array[np.isin(array, ('skt', 'lgt', 'kt', 'Unknown'), invert=True)] = '잡캐리어'
    request['carrier'] = array

    array = np.array(request['region'], dtype='<U16')
    array[np.isin(array, feature_stat['most_freq_region'], invert=True)] = '잡지역'
    request['region'] = array

    array = np.array(request['device_name'], dtype='<U16')
    array[np.isin(array, feature_stat['most_freq_device'], invert=True)] = '잡디바이스'
    request['device_name'] = array

    # Preprocess Historical Features
    for feature in ['viewer_avg_ctr', 'lineitem_avg_ctr']:
        array = np.array(request[feature])
        np.nan_to_num(array, copy=False, nan=feature_stat[feature])
        request[feature] = array

    for feature in ['viewer_imp_count', 'viewer_clk_count', 'lineitem_imp_count', 'lineitem_clk_count']:
        array = np.array(request[feature])
        np.nan_to_num(array, copy=False, nan=0.0)
        request[feature] = array

    # Scale
    scaled_array = np.stack([request[feature] for feature in ['viewer_avg_ctr','viewer_imp_count',
                                                              'lineitem_avg_ctr','viewer_clk_count',
                                                              'lineitem_clk_count','lineitem_imp_count',
                                                              'age','registered_days']], axis=1)
    scaled_array = scaler.transform(scaled_array)

    # Hash
    hashed_array = hasher.transform(np.array(request['ip'])).toarray()

    # 1H Encode
    encoded_array = np.stack([request[feature] for feature in ['sex', 'network_type', 'carrier',
                                                               'region', 'device_name','creative_name',
                                                               'lineitem_id', 'ad_group_id', 'lineitem_category',
                                                               'unit_id', 'app_id', 'organization_id',
                                                               'impressed_at_hour', 'impressed_at_weekday']], axis=1)
    encoded_array = encoded_array.astype('object') # TODO: Does this influence the results?
    encoded_array = encoder.transform(encoded_array)

    final_array = np.concatenate([scaled_array, hashed_array, encoded_array], axis=1)

    result = lr.predict_proba(final_array)[:, 1:].squeeze().tolist()
    return result

@app.get('/')
def get_root():
    return {'message': 'Welcome to the spam detection API'}

@app.get('/get_inference_scores/')
async def get_inference_scores():
    return {'scores': getinferencescores()}

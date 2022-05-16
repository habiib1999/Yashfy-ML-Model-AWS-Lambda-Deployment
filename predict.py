import numpy as np
import pandas as pd 
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
#import transformers
from transformers import AutoModel, BertTokenizerFast, AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification , TFBertModel, TFAutoModel
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping ,ReduceLROnPlateau
from arabert.preprocess import ArabertPreprocessor 
from keras.models import load_model
import keras.backend as K

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


#review = ['العيادة مناسبة ولكن السعر مرتفع','طاقم العمل غير جيد و لكن العيادة رائعة']
#review = ['دكتور ممتاز و العيادة نظيفة ولكن السعر مرتفع']

def map_bert(inputs):
    inputs = {'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']}
    return inputs

def predict(review):
    model_name = 'aubmindlab/bert-base-arabertv02'
    model = load_model('/opt/ml/model/FullModel.h5',custom_objects={"TFBertModel": TFBertModel , "f1_metric": f1_metric})

    preprocessed_review = []
    arabert_prep = ArabertPreprocessor(model_name=model_name)
    preprocessed_review.append(arabert_prep.preprocess(review[0]))
    preprocessed_review.append(arabert_prep.preprocess(review[1]))

    arabert_tokenizer = BertTokenizerFast.from_pretrained(model_name)
    review_token = arabert_tokenizer(preprocessed_review, truncation=True, padding='max_length', max_length=40)
    review = tf.data.Dataset.from_tensor_slices(review_token)
    print(review)
    review = review.map(map_bert)


    prediction = model.predict({'input_ids': tf.stack( tfds.as_dataframe(review)['input_ids'].values )
    ,'attention_mask' : tf.stack( tfds.as_dataframe(review)['attention_mask'].values )})

    test_predictions_df = {'Clinic': np.argmax(prediction[0],axis = 1), 'Doctor': np.argmax(prediction[1],axis = 1)
        , 'Staff': np.argmax(prediction[2],axis = 1), 'Time': np.argmax(prediction[3],axis = 1)
        , 'Equipments': np.argmax(prediction[4],axis = 1), 'Price': np.argmax(prediction[5],axis = 1)}
    test_predictions_df = pd.DataFrame(data=test_predictions_df)
    test_predictions_df['Clinic'] = test_predictions_df['Clinic'].map({0: -1, 1: 0, 2:1})
    test_predictions_df['Doctor'] = test_predictions_df['Doctor'].map({0: -1, 1: 0, 2:1})
    test_predictions_df['Staff'] = test_predictions_df['Staff'].map({0: -1, 1: 0, 2:1})
    test_predictions_df['Time'] = test_predictions_df['Time'].map({0: -1, 1: 0, 2:1})
    test_predictions_df['Equipments'] = test_predictions_df['Equipments'].map({0: -1, 1: 0, 2:1})
    test_predictions_df['Price'] = test_predictions_df['Price'].map({0: -1, 1: 0, 2:1})
    print(test_predictions_df)
    return test_predictions_df



def lambda_handler(event, context):
    data = json.loads(event['body'])
    review = data['review']
    prediction = predict(review)
    print(prediction)
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(
            {
                "predicted_label": prediction.to_json(),
            }
        )
    }

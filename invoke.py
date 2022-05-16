import requests
import json
import numpy as np

data = {
    'review': ['العيادة مناسبة ولكن السعر مرتفع','طاقم العمل غير جيد و لكن العيادة رائعة']
}

headers = {
    'Content-type': "application/json"
}
# Main code for post HTTP request
url = "http://127.0.0.1:3000/predict"
response = requests.request("POST", url, headers=headers, data=json.dumps(data))

# Show confusion matrix and display accuracy
lambda_predictions = np.array(response.json())
print(lambda_predictions)
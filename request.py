
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Age':20, 'Subscription_Length_Months':10, 'Monthly_Bill':42, 'Total_Usage_GB':150}, 'Gender':0)

print(r.json())



{
  "year": 2019,
  "kms": 42000,
  "owner": 1,
  "fuel": "Petrol",
  "seller": "Individual",
  "transmission": "Manual",
  "location": "Delhi"
}
import requests

url = "http://127.0.0.1:8000/predict"

data = {
  "year": 2019,
  "kms": 42000,
  "owner": 1,
  "fuel": "Petrol",
  "seller": "Individual",
  "transmission": "Manual",
  "location": "Delhi"
}

response = requests.post(url, json=data)
print(response.json())

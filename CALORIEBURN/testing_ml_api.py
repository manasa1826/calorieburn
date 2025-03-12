import json
import requests

input_data={
  "Gender": 0,
  "Age": 0,
  "Height":0 ,
  "Weight": 0,
  "Duration": 0,
  "Heart_Rate": 0,
  "Body_Temp": 0
}

url="http://127.0.0.1:8081/calorie_burn"
json_object=json.dumps(input_data)
response=requests.post(url,data=json_object)
print(response.text)
import requests
import json
import requests

'''
objective: to test the api.
'''
url = "https://your url/is-fault"  #to test in production.
#url='http://127.0.0.1:8000/is-fraud'  #-- to test in local host
r = requests.post(url, 
json = {
"sensor1":0.0,
"sensor2":0.0,
"sensor3":0.0,
"sensor4":0.0,
})


print("------",r)
print("------",r.text)
import requests
import os


ORG = 'iiithf'
QUERY = ''
try: QUERY = os.environ['QUERY']
except: None


r = requests.post('http://%s?sql=SELECT * FROM "%s" WHERE "id" LIKE \'%%input%%\'', (QUERY, ORG))
inp = r.json()[0]
r = requests.post('http://%s?sql=SELECT * FROM "%s" WHERE "id" LIKE \'%%model%%\'', (QUERY, ORG))
mod = r.json()[0]
print('inp', inp)
print('mod', mod)

import requests
import os


ORG = 'iiithf'
QUERY = ''
try: QUERY = os.environ['QUERY']
except: None

data = {'sql': 'SELECT * FROM "{}" WHERE "id" LIKE \'%%input%%\''.format(ORG)}
r = requests.get('http://{}/{}'.format(QUERY, ORG), data=data)
inps = r.json()
data = {'sql': 'SELECT * FROM "{}" WHERE "id" LIKE \'%%model%%\''.format(ORG)}
r = requests.get('http://{}/{}'.format(QUERY, ORG), data=data)
mods = r.json()
print('inp', inps)
print('mod', mods)

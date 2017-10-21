import json

with open('/home/pi/Zavrsni/config.json') as f:
    string = f.read()

dictionary = json.loads(string)

conf = type('test', (object,), {})()

conf.__dict__ = dictionary

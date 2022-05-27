import requests

# Create a dictionary to be sent.
json_data = {'name':'sidserver', 'length':2}

# Send the data.
response = requests.post(url='http://192.168.0.156:5000/cmd/ext', json=json_data)
print("Server responded with %s" % response.status_code)
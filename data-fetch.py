import requests
import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
apikey = os.getenv("API_KEY")

baseurl = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
params = {
    "api-key": apikey,
    "format": "json",
    "limit": 1
}
response = requests.get(baseurl, params=params)
data = response.json()
totalcount = data['total']
print(totalcount)

new_response = requests.get(baseurl, params={"api-key": apikey, "format":"json", "limit": totalcount})
new_data = new_response.json()

#filtering the records -> state, commodity, arrival_data, modal_price
filtered_data = []
for rec in new_data['records']:
    filtered_data.append({
        "state": rec['state'], 
        "commodity": rec['commodity'],
        "arrival_date": rec['arrival_date'],
        "modal_price": rec['modal_price']
    })

#saving filtered data to a csv
df = pd.DataFrame(filtered_data)
df.to_csv("data/mandi_prices.csv", index=False)


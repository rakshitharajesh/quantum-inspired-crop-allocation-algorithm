import os
from dotenv import load_dotenv
import requests
import pandas as pd
load_dotenv()
apikey = os.getenv("API_KEY")
print(apikey)
baseurl = f"https://api.data.gov.in/resource/35be999b-0208-4354-b557-f6ca9a5355de"
params = {
    "api-key": apikey,
    "format": "json",
    "limit": 1
}
response = requests.get(baseurl, params=params)
data = response.json()
totalcount = data['total']
print(totalcount)

new_response = requests.get(baseurl, params={"api-key": apikey, "format": "json", "limit": totalcount})
new_data = new_response.json()

#filter the records -> state, commodity
filtered_data = []
for rec in new_data['records']:
    filtered_data.append({
        "state_name": rec['state_name'],
        "crop": rec['crop'],
        "yield": rec['production_'],
        "area_": rec['area_'],
        "crop_year": rec['crop_year']
    })
#saving the filtered data to a csv
df = pd.DataFrame(filtered_data)
df.to_csv("data/yield_data.csv", index=False)

# note: yield/area_ should give the actual yield
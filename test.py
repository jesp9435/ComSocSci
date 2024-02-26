import requests
import pandas as pd
import concurrent.futures 
import requests
import math
f=open("authors.txt","r",encoding="utf-8")
url="https://api.openalex.org/works?per-page=200&filter=author.id:https://openalex.org/A5035210768"
r=requests.get(url).json()
print(r)
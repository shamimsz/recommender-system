import requests
from zipfile import ZipFile
import os

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
FILE = "ml-100k.zip"

response = requests.get(url= URL, stream= True)
response.raise_for_status()

with open(file= FILE, mode= "wb") as zfile:
    for chunk in response.iter_content(chunk_size= 128):
        zfile.write(chunk)

with ZipFile(file= FILE) as file:
    file.extractall("data/")

os.remove(FILE)

print("Done! Data extracted to data/ml-100k/")

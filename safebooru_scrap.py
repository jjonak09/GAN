import requests
import re
import uuid
from bs4 import BeautifulSoup
import time

url = "https://safebooru.org/index.php?page=post&s=list&tags=white_background"

for page in range(600):
    print(page)
    time.sleep(1)
    if page == 0:
        r = requests.get(url)
    else:
        r = requests.get(url + '&pid=' + str(40 * page))

    soup = BeautifulSoup(r.text, 'lxml')
    imgs = soup.find_all('img', src=re.compile(
        '^//safebooru.org/thumbnails/'))

    for img in imgs:
        # print(img['src'])
        img_url = img['src'].replace('thumbnails', '/images')
        img_url = img_url.replace('thumbnail_', '')
        # print('https:' + img_url)
        r = requests.get('https:' + img_url)
        with open(str('./image/') + str(uuid.uuid4()) + str('.jpg'), 'wb') as file:
            file.write(r.content)

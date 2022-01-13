import requests
import os
from unsplash.api import Api
from unsplash.auth import Auth

headers = {"Accept": "*/*", "Accept-Encoding": "gzip, deflate, br", "Accept-Language": "en-US,en;q=0.5",
                        "Connection": "keep-alive",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:81.0) Gecko/20100101 Firefox/81.0"}


client_id = "jwMlQ4kdF-VHLvhb-zEGwJKxD67tBI7HP7qEM6hAWwY"
client_secret = "2_VOsednZ_PA_0Sh6X6slxOjs2YeH3W6IchxEF7Oqck"
redirect_uri = ""
code = ""

auth = Auth(client_id, client_secret, redirect_uri, code=code)
api = Api(auth)

download_dir = "unsplash"
if not os.path.exists(download_dir):
    os.mkdir(download_dir)

# The API limit is 50 requests per hour
# But
for _ in range(50):
    photos = api.photo.random(count=30, w=1280, h=720, orientation='landscape')
    for photo in photos:
        name = photo.id
        url = photo.urls.regular

        filepath = f"{os.path.join(os.path.realpath(os.getcwd()), download_dir, name)}.jpg"

        with open(filepath, "wb") as f:
            f.write(requests.request("GET", url, headers=headers).content)





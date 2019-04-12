from PIL import Image
from io import BytesIO
import numpy as np
import requests
import base64

img = Image.open("image.jpg")

buffer = BytesIO()
img.save(buffer, "JPEG")
img_str = base64.b64encode(buffer.getvalue())

host = "localhost"
for i in range(4):
    try:
        r = requests.get("http://"+host+":7{0:03}/predict/".format(i), 
            params={"data":img_str})
    except Exception as e:
        print("Error with API in port 7{0:03}".format(i))
        continue

    status = r.status_code
    if status != 200:
        print("Error with API in port 7{0:03}".format(i))
        continue

    print(r.json())
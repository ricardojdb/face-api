from PIL import Image
from io import BytesIO
import numpy as np
import requests
import base64
import traceback
import sys
import json

img = Image.open("image.jpg")

buffer = BytesIO()
img.save(buffer, "JPEG")
img_str = base64.b64encode(buffer.getvalue())

host = "localhost"
for i in range(3, 7):
    try:
        r = requests.post(
            url="http://"+host+":7{0:03}/predict".format(i),
            data=img_str,
            timeout=5)
    except Exception as e:
        # traceback.print_exc(file=sys.stdout)
        print("Error with API in port 7{0:03}".format(i))
        continue

    status = r.status_code
    if status != 200:
        print("Error with API in port 7{0:03}".format(i))
        continue

    print(r.json())

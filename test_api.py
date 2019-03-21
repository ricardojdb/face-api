from PIL import Image
from io import BytesIO
import numpy as np
import requests
import base64

img = Image.open("image.jpg")

buffer = BytesIO()
img.save(buffer, "JPEG")
img_str = base64.b64encode(buffer.getvalue())

for i in range(4):
	r = requests.get("http://192.168.8.100:7{0:03}/predict/".format(i), data=img_str)

	status = r.status_code
	if status != 200:
		print("Error with API in port 7{0:03}".format(i))
		break

	print(r.json())
"""
Simple attack simulator
"""

import requests
import time

URLS = [
    "http://localhost:8000/search?q=' OR 1=1--",
    "http://localhost:8000/test?<script>alert(1)</script>",
    "http://localhost:8000/../../etc/passwd",
]

while True:
    for url in URLS:
        try:
            r = requests.get(url)
            print(f"{url} -> {r.status_code}")
        except Exception as e:
            print(e)

        time.sleep(1)

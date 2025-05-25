import requests
import os
import pandas as pd
from datetime import datetime

# Configuration
USERNAME = 'BrianX'  # AmeriFlux login email
PASSWORD = 'D@bFyZF6paM9qmu'  # AmeriFlux password
SITE_ID = 'US-CRT'  # AmeriFlux site code
OUTPUT_DIR = './ameriflux_data'

LOGIN_URL = 'https://ameriflux.lbl.gov/AmeriFlux/login'
DOWNLOAD_URL = f'https://ameriflux.lbl.gov/AmeriFlux/data/download/datafile/BASE-BADM-{SITE_ID}.zip'

# Start session
session = requests.Session()

def login():
    """Logs in to AmeriFlux to gain access to download."""
    payload = {
        'email': USERNAME,
        'password': PASSWORD,
        'remember': 'on'
    }
    resp = session.post(LOGIN_URL, data=payload)
    if "Logout" not in resp.text:
        raise Exception("Login failed – check credentials or site structure.")
    print("✅ Logged in successfully.")

def download_data():
    """Downloads AmeriFlux data after logging in."""
    login()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_file = os.path.join(OUTPUT_DIR, f'{SITE_ID}_BASE_{timestamp}.zip')

    print(f"⬇ Downloading data to {output_file}")
    r = session.get(DOWNLOAD_URL)
    if r.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(r.content)
        print("✅ Download complete.")
    else:
        print("❌ Failed to download:", r.status_code)

# Run script
if __name__ == '__main__':
    download_data()

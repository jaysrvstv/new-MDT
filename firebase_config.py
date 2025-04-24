import firebase_admin
from firebase_admin import credentials, firestore

# Replace with the path to your Firebase Admin SDK JSON key
SERVICE_ACCOUNT_KEY_PATH = "your-key.json"

# Initialize Firebase app
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client()

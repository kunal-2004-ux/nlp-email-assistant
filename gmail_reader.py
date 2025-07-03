import os
import base64
import pickle
import time
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    """Authenticate with Gmail API using OAuth 2.0"""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def get_latest_emails(service, max_results=50):
    """Fetch latest emails from Gmail inbox"""
    try:
        results = service.users().messages().list(
            userId='me', 
            labelIds=['INBOX'], 
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])
        emails = []

        for msg in messages:
            msg_data = service.users().messages().get(
                userId='me', 
                id=msg['id'],
                format='full'  # Get full message data for better processing
            ).execute()
            
            headers = msg_data['payload']['headers']
            
            # Extract email metadata with improved handling
            subject = next((header['value'] for header in headers if header['name'].lower() == 'subject'), '(No Subject)')
            sender = next((header['value'] for header in headers if header['name'].lower() == 'from'), 'Unknown')
            date = next((header['value'] for header in headers if header['name'].lower() == 'date'), '')
            
            # Extract email body with better MIME type handling
            body = ""
            if 'parts' in msg_data['payload']:
                for part in msg_data['payload']['parts']:
                    if part.get('mimeType') == 'text/plain' and 'data' in part.get('body', {}):
                        body = base64.urlsafe_b64decode(part['body']['data'].encode('UTF-8')).decode('utf-8', errors='ignore')
                        break
                    elif part.get('mimeType') == 'text/html' and 'data' in part.get('body', {}):
                        body_html = base64.urlsafe_b64decode(part['body']['data'].encode('UTF-8')).decode('utf-8', errors='ignore')
                        soup = BeautifulSoup(body_html, 'html.parser')
                        body = soup.get_text()
                        break
            elif 'data' in msg_data['payload'].get('body', {}):
                body = base64.urlsafe_b64decode(msg_data['payload']['body']['data'].encode('UTF-8')).decode('utf-8', errors='ignore')

            if not body.strip():
                body = "(No content)"

            emails.append({
                'id': msg['id'],
                'subject': subject,
                'sender': sender,
                'date': date,
                'body': body.strip(),
                'labels': msg_data.get('labelIds', [])
            })

        return emails

    except Exception as e:
        print(f"Error fetching emails: {e}")
        return []

def get_email_by_id(service, email_id):
    """Fetch a specific email by its ID"""
    try:
        msg_data = service.users().messages().get(userId='me', id=email_id).execute()
        headers = msg_data['payload']['headers']
        
        subject = next((header['value'] for header in headers if header['name'].lower() == 'subject'), '(No Subject)')
        sender = next((header['value'] for header in headers if header['name'].lower() == 'from'), 'Unknown')
        date = next((header['value'] for header in headers if header['name'].lower() == 'date'), '')

        if 'data' in msg_data['payload']['body']:
            body_data = msg_data['payload']['body']['data']
        else:
            parts = msg_data['payload'].get('parts', [])
            body_data = ''
            for part in parts:
                if 'body' in part and 'data' in part['body']:
                    body_data = part['body']['data']
                    break

        body = base64.urlsafe_b64decode(body_data.encode('UTF-8')).decode('utf-8', errors='ignore')
        soup = BeautifulSoup(body, 'html.parser')
        text = soup.get_text()

        return {
            'id': email_id,
            'subject': subject,
            'sender': sender,
            'date': date,
            'body': text.strip()
        }

    except Exception as e:
        print(f"Error fetching email {email_id}: {e}")
        return None 
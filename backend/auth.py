import time
import jwt

SECRET_KEY = 'fortipay_secret_key'
ALGORITHM = 'HS256'
USERS = {
    'admin': 'password123',
    'analyst': 'fraudbuster'
}

def authenticate_user(username, password):
    return USERS.get(username) == password

def generate_jwt(payload, expires_in=3600):
    payload = payload.copy()
    payload['exp'] = int(time.time()) + expires_in
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_jwt(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except Exception:
        return None 
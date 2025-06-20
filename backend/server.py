import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from auth import authenticate_user, generate_jwt, verify_jwt
from fraud import predict_fraud, explain_alert
from storage import save_transaction, get_transaction, get_risk_score

class RequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def _parse_json(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return None
        post_data = self.rfile.read(content_length)
        try:
            return json.loads(post_data)
        except Exception:
            return None

    def _unauthorized(self):
        self._set_headers(401)
        self.wfile.write(json.dumps({'error': 'Unauthorized'}).encode())

    def _require_auth(self):
        auth_header = self.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            self._unauthorized()
            return None
        token = auth_header.split(' ')[1]
        payload = verify_jwt(token)
        if not payload:
            self._unauthorized()
            return None
        return payload

    def do_POST(self):
        if self.path == '/auth/login':
            data = self._parse_json()
            if not data or 'username' not in data or 'password' not in data:
                self._set_headers(400)
                self.wfile.write(json.dumps({'error': 'Missing credentials'}).encode())
                return
            if authenticate_user(data['username'], data['password']):
                token = generate_jwt({'username': data['username']})
                self._set_headers(200)
                self.wfile.write(json.dumps({'token': token}).encode())
            else:
                self._set_headers(401)
                self.wfile.write(json.dumps({'error': 'Invalid credentials'}).encode())
        elif self.path == '/transaction/submit':
            user = self._require_auth()
            if not user:
                return
            data = self._parse_json()
            if not data:
                self._set_headers(400)
                self.wfile.write(json.dumps({'error': 'Invalid transaction data'}).encode())
                return
            tx_id = save_transaction(data)
            risk_score, prediction = predict_fraud(data)
            explanation = explain_alert(data, risk_score, prediction)
            self._set_headers(200)
            self.wfile.write(json.dumps({'transaction_id': tx_id, 'risk_score': risk_score, 'prediction': prediction, 'explanation': explanation}).encode())
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'Not found'}).encode())

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith('/transaction/risk/'):
            user = self._require_auth()
            if not user:
                return
            tx_id = parsed.path.split('/')[-1]
            tx = get_transaction(tx_id)
            if not tx:
                self._set_headers(404)
                self.wfile.write(json.dumps({'error': 'Transaction not found'}).encode())
                return
            risk_score, prediction = get_risk_score(tx_id)
            self._set_headers(200)
            self.wfile.write(json.dumps({'transaction_id': tx_id, 'risk_score': risk_score, 'prediction': prediction}).encode())
        elif parsed.path.startswith('/alerts/'):
            user = self._require_auth()
            if not user:
                return
            tx_id = parsed.path.split('/')[-1]
            tx = get_transaction(tx_id)
            if not tx:
                self._set_headers(404)
                self.wfile.write(json.dumps({'error': 'Transaction not found'}).encode())
                return
            risk_score, prediction = get_risk_score(tx_id)
            explanation = explain_alert(tx, risk_score, prediction)
            self._set_headers(200)
            self.wfile.write(json.dumps({'transaction_id': tx_id, 'explanation': explanation}).encode())
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'Not found'}).encode())

def run(server_class=HTTPServer, handler_class=RequestHandler):
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    print('Starting server on port 8000...')
    httpd.serve_forever()

if __name__ == '__main__':
    run() 
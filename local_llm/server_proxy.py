# local_server.py - Runs on your home network
from flask import Flask, request, jsonify, render_template, send_from_directory
import requests
import threading
import time
import json
import uuid
import os

app = Flask(__name__)

# Configuration
RELAY_SERVER_URL = "https://swcmakeendpoints.uc.r.appspot.com/"
LOCAL_SERVER_ID ="TheReallyAmazingAI88RelayServer!"# str(uuid.uuid4())  # Unique ID for this server instance

# Your actual web application routes
@app.route('/')
def home():
    return render_template('proxy_home.html', server_id=LOCAL_SERVER_ID)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/api/data')
def api_data():
    return jsonify({"message": "Hello from local server!", "timestamp": time.time()})

# Function to poll for requests from relay server
def poll_relay_server():
    while True:
        try:
            print(f"Polling relay server for requests...")
            # Long-polling request to relay server
            response = requests.get(
                f"{RELAY_SERVER_URL}/get_request/{LOCAL_SERVER_ID}",
                timeout=30  # 30-second timeout for long-polling
            )
            
            print(f"Poll response status: {response.status_code}")
            
            if response.status_code == 200:
                request_data = response.json()
                print(f"Received data: {request_data}")
                
                if request_data:
                    print(f"Processing request: {request_data.get('method')} {request_data.get('path')}")
                    # Process the request
                    result = handle_proxied_request(request_data)
                    
                    # Send response back to relay server
                    send_response = requests.post(
                        f"{RELAY_SERVER_URL}/send_response/{LOCAL_SERVER_ID}",
                        json={
                            "request_id": request_data["request_id"],
                            "response": result
                        }
                    )
                    print(f"Response sent back, status: {send_response.status_code}")
                else:
                    print("No pending requests")
            else:
                print(f"Poll failed with status: {response.status_code}")
            
        except requests.exceptions.Timeout:
            # Timeout is expected with long-polling, just continue
            print("Poll timeout (normal)")
            continue
        except Exception as e:
            print(f"Error polling relay server: {e}")
            time.sleep(5)  # Wait before retrying

def handle_proxied_request(request_data):
    """Handle a request that came through the relay server"""
    method = request_data.get("method", "GET")
    path = request_data.get("path", "/")
    headers = request_data.get("headers", {})
    body = request_data.get("body", "")
    
    print(f"Handling proxied request: {method} {path}")
    print(f"Headers: {headers}")
    
    # Make internal request to our Flask app
    try:
        with app.test_client() as client:
            # Remove problematic headers that might interfere
            clean_headers = {}
            for key, value in headers.items():
                if key.lower() not in ['host', 'content-length', 'connection']:
                    clean_headers[key] = value
            
            if method == "GET":
                response = client.get(path, headers=clean_headers)
            elif method == "POST":
                response = client.post(path, data=body, headers=clean_headers)
            elif method == "PUT":
                response = client.put(path, data=body, headers=clean_headers)
            elif method == "DELETE":
                response = client.delete(path, headers=clean_headers)
            else:
                response = client.get(path, headers=clean_headers)
            
            print(f"Response status: {response.status_code}")
            print(f"Response content type: {response.content_type}")
            
            result = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.get_data(as_text=True),
                "content_type": response.content_type
            }
            
            print(f"Returning response with {len(result['body'])} characters")
            return result
            
    except Exception as e:
        print(f"Error handling proxied request: {e}")
        return {
            "status_code": 500,
            "headers": {"Content-Type": "text/plain"},
            "body": f"Internal server error: {str(e)}",
            "content_type": "text/plain"
        }

def register_with_relay():
    """Register this server with the relay"""
    try:
        response = requests.post(
            f"{RELAY_SERVER_URL}/register_server",
            json={"server_id": LOCAL_SERVER_ID}
        )
        if response.status_code == 200:
            print(f"Successfully registered with relay server. Server ID: {LOCAL_SERVER_ID}")
        else:
            print(f"Failed to register with relay server: {response.status_code}")
    except Exception as e:
        print(f"Error registering with relay server: {e}")

if __name__ == '__main__':
    # Register with relay server
    register_with_relay()
    
    # Start polling thread
    polling_thread = threading.Thread(target=poll_relay_server, daemon=True)
    polling_thread.start()
    
    # Start Flask server (for local development/testing)
    print(f"Local server starting with ID: {LOCAL_SERVER_ID}")
    print(f"Your public URL will be: {RELAY_SERVER_URL}/proxy/{LOCAL_SERVER_ID}")
    app.run(host='0.0.0.0', port=5000, debug=False)
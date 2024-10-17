from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from datetime import datetime, timedelta
from vvn_controller import create_prompt, generate, send_continue_chat, send_new_chat
from model import db, ChatHistory
from dotenv import load_dotenv
import ssl
import os

app = Flask(__name__)
CORS(app)  # Enable CORS
socketio = SocketIO(app, cors_allowed_origins='*')

# Load environment variables from .env file
load_dotenv()

def create_response(data, status_code, message):
    return jsonify({
        'data': data,
        'statusCode': status_code,
        'message': message
    }), status_code

@app.route('/chat', methods=['POST'])
def send_message():
    try:
        data = request.json
        user_query = data.get('content')

        if (user_query is not None) and (user_query != ''):
            response = send_new_chat(user_query)

            return create_response(response, 200, 'Success')
        return create_response(None, 400, 'No query or userId provided')
    except Exception as e:
        print(e)
        return create_response(None, 500, f'Internal Server Error: {str(e)}')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=4646)


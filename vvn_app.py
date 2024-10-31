import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from vvn_controller import  send_continue_chat
from model import ChatHistory
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)  # Enable CORS

BACKEND_URL = "https://shop-ecommerce-be.onrender.com/chat/get-messages"


# Load environment variables from .env file
load_dotenv()

def create_response(data, status_code, message):
    return jsonify({
        'data': data,
        'statusCode': status_code,
        'message': message
    }), status_code

def get_chat_history(chatId):
    try:
        response = requests.get(f"{BACKEND_URL}/{chatId}")
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        if data['statusCode'] == 200:
            history = []
            if data['data']:
                for res in data['data']:
                    if res['isUser']:
                        history.append(ChatHistory(content=res['content'], is_user=True))
                    else:
                        history.append(ChatHistory(content=res['content'], is_user=False))
            return history
        else:
            raise Exception(f"Error: {data['message']}")
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

@app.route('/chat', methods=['POST'])
def send_message():
    try:
        data = request.json
        user_query = data['content']
        user_id = data['userId']
        chat_id = data['chatId']

        if user_query and user_id and chat_id:
            chat_history = get_chat_history(chat_id)
            if chat_history:
                answer = send_continue_chat(chat_history, user_query)
            else:
                answer = send_continue_chat([], user_query)
            return create_response(answer, 200, 'Success')

        return create_response(None, 400, 'No query or userId or chatId provided')
    except Exception as e:
        print(e)
        return create_response(None, 500, f'Internal Server Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4646)

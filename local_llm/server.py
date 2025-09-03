from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import sys,os
# Get the directory containing qwen_.py
qwen_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, qwen_dir)

from qwen_ import Qwen

import threading
import json
import socket
import platform
from typing import Dict, Any


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key-for-qwen'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global Qwen instance
qwen_instance = None

def get_machine_ip():
    """Get the current machine's IP address"""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback to localhost if unable to determine IP
        return "127.0.0.1"

def print_host_details(host_ip, port):
    """Print detailed host information"""
    print("=" * 60)
    print("LLM Flask Server - Host Details")
    print("=" * 60)
    print(f"Machine Name: {platform.node()}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python Version: {platform.python_version()}")
    print("-" * 60)
    print(f"Server Host IP: {host_ip}")
    print(f"Server Port: {port}")
    print(f"Server URL: http://{host_ip}:{port}")
    print("-" * 60)
    
    # Additional network information
    try:
        hostname = socket.gethostname()
        fqdn = socket.getfqdn()
        print(f"Hostname: {hostname}")
        print(f"FQDN: {fqdn}")
        
        # Get all network interfaces
        print("\nNetwork Interfaces:")
        import netifaces
        interfaces = netifaces.interfaces()
        for interface in interfaces:
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr in addrs[netifaces.AF_INET]:
                    print(f"  {interface}: {addr['addr']}")
    except ImportError:
        # netifaces not available, skip detailed network info
        print("Install 'netifaces' for detailed network interface information")
    except Exception as e:
        print(f"Error getting network details: {e}")
    
    print("=" * 60)

def initialize_qwen():
    """Initialize the Qwen instance at server startup"""
    global qwen_instance
    try:
        qwen_instance = Qwen()
        print("Qwen instance initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing Qwen: {e}")
        return False

@app.route('/')
def index():
    """Serve the main HTML template"""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    # Send initial status
    emit('status', {
        'connected': True,
        'qwen_ready': qwen_instance is not None,
        'message': 'Connected to LLM server'
    })
    
    # Send current token stats
    if qwen_instance:
        stats = qwen_instance.get_token_stats()
        emit('token_stats', stats)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('submit_input')
def handle_submit_input(data):
    """Handle user input submission with streaming response"""
    if not qwen_instance:
        emit('error', {'message': 'Qwen instance not initialized'})
        return
    
    try:
        user_input = data.get('message', '')
        max_tokens = data.get('max_tokens', 512)
        auto_execute_tools = data.get('auto_execute_tools', True)
        use_history = data.get('use_message_history', True)
        
        if not user_input.strip():
            emit('error', {'message': 'Empty input provided'})
            return
        
        print(f"Processing input: {user_input}")

        #res = qwen_instance.generate_response(user_input)
        #print(res)  
        
        # Get the current session ID to maintain context in background thread
        session_id = request.sid
        
        # Start streaming response in a separate thread
        def stream_worker():
            try:
                # Emit with explicit session ID to maintain context
                socketio.emit('stream_start', {'message': 'Starting response generation...'}, to=session_id)
                
                response_parts = []
                for token in qwen_instance.stream_response(
                    user_input=user_input,
                    max_new_tokens=max_tokens,
                    auto_execute_tools=auto_execute_tools,
                    use_message_history=use_history,
                    print_tokens=False  # We'll handle token emission via websocket
                ):
                    response_parts.append(token)
                    #print(token)
                    
                    # Emit each token as it's generated with session ID
                    socketio.emit('stream_token', {'token': token}, to=session_id)
                
                # Complete response
                complete_response = "".join(response_parts)
                print(complete_response)
                # Send completion signal with full response
                socketio.emit('stream_complete', {
                    'complete_response': complete_response,
                    'message': 'Response generation completed'
                }, to=session_id)
                
                # Send updated token stats
                stats = qwen_instance.get_token_stats()
                socketio.emit('token_stats', stats, to=session_id)
                
            except Exception as e:
                print(f"Error during streaming: {e}")
                socketio.emit('error', {'message': f'Error generating response: {str(e)}'}, to=session_id)
        
        # Start streaming in background thread
        thread = threading.Thread(target=stream_worker)
        thread.daemon = True
        thread.start()
        
    except Exception as e:
        print(f"Error handling input: {e}")
        emit('error', {'message': f'Error processing input: {str(e)}'})

@socketio.on('undo_last_message')
def handle_undo_last_message():
    """Handle undo last message request"""
    if not qwen_instance:
        emit('error', {'message': 'Qwen instance not initialized'})
        return
    
    try:
        qwen_instance.clear_last_message()
        print("Last message removed")
        
        emit('message_undone', {'message': 'Last message removed successfully'})
        
        # Send updated token stats
        stats = qwen_instance.get_token_stats()
        emit('token_stats', stats)
        
    except Exception as e:
        print(f"Error undoing last message: {e}")
        emit('error', {'message': f'Error undoing last message: {str(e)}'})

@socketio.on('change_system_prompt')
def handle_change_system_prompt(data):
    """Handle system prompt change request"""
    if not qwen_instance:
        emit('error', {'message': 'Qwen instance not initialized'})
        return
    
    try:
        new_prompt = data.get('system_prompt', '')
        
        if not new_prompt.strip():
            emit('error', {'message': 'Empty system prompt provided'})
            return
        
        qwen_instance._update_system_prompt(new_prompt)
        print(f"System prompt updated to: {new_prompt}")
        
        emit('system_prompt_updated', {
            'message': 'System prompt updated successfully',
            'new_prompt': new_prompt
        })
        
    except Exception as e:
        print(f"Error updating system prompt: {e}")
        emit('error', {'message': f'Error updating system prompt: {str(e)}'})

@socketio.on('clear_chat')
def handle_clear_chat():
    """Handle clear chat messages request"""
    if not qwen_instance:
        emit('error', {'message': 'Qwen instance not initialized'})
        return
    
    try:
        qwen_instance.clear_chat_messages()
        print("Chat messages cleared")
        
        emit('chat_cleared', {'message': 'Chat history cleared successfully'})
        
        # Send updated token stats
        stats = qwen_instance.get_token_stats()
        emit('token_stats', stats)
        
    except Exception as e:
        print(f"Error clearing chat: {e}")
        emit('error', {'message': f'Error clearing chat: {str(e)}'})

@socketio.on('auto_append_conversation')
def handle_auto_append_conversation():
    """Handle auto appending to chat messages request"""
    if not qwen_instance:
        emit('error', {'message': 'Qwen instance not initialized'})
        return
    
    try:
        if qwen_instance.auto_append_conversation:
            qwen_instance.auto_append_conversation = False
            emit('auto_append_convo', {'message': False})
        else:
            qwen_instance.auto_append_conversation = True
            emit('auto_append_convo', {'message': True})
        
    except Exception as e:
        print(f"Error enabling auto append chat: {e}")
        emit('error', {'message': f'Error clearing chat: {str(e)}'})

@socketio.on('get_token_stats')
def handle_get_token_stats():
    """Handle request for current token statistics"""
    if not qwen_instance:
        emit('error', {'message': 'Qwen instance not initialized'})
        return
    
    try:
        stats = qwen_instance.get_token_stats()
        emit('token_stats', stats)
    except Exception as e:
        print(f"Error getting token stats: {e}")
        emit('error', {'message': f'Error getting token stats: {str(e)}'})

@socketio.on('list_tools')
def handle_list_tools():
    """Handle request to list available tools"""
    if not qwen_instance:
        emit('error', {'message': 'Qwen instance not initialized'})
        return
    
    try:
        tools = qwen_instance.list_available_tools()
        emit('tools_list', {'tools': tools})
    except Exception as e:
        print(f"Error listing tools: {e}")
        emit('error', {'message': f'Error listing tools: {str(e)}'})

@socketio.on('remove_tool')
def handle_remove_tool(data):
    """Handle tool removal request"""
    if not qwen_instance:
        emit('error', {'message': 'Qwen instance not initialized'})
        return
    
    try:
        tool_name = data.get('tool_name', '')
        if not tool_name:
            emit('error', {'message': 'Tool name not provided'})
            return
        
        success = qwen_instance.remove_tool(tool_name)
        if success:
            emit('tool_removed', {
                'message': f'Tool "{tool_name}" removed successfully',
                'tool_name': tool_name
            })
        else:
            emit('error', {'message': f'Tool "{tool_name}" not found'})
            
    except Exception as e:
        print(f"Error removing tool: {e}")
        emit('error', {'message': f'Error removing tool: {str(e)}'})

@socketio.on('server_status')
def handle_server_status():
    """Handle server status request"""
    status = {
        'server_running': True,
        'qwen_initialized': qwen_instance is not None,
        'available_endpoints': [
            'submit_input',
            'undo_last_message', 
            'change_system_prompt',
            'clear_chat',
            'get_token_stats',
            'list_tools',
            'remove_tool'
        ]
    }
    emit('server_status', status)

if __name__ == '__main__':
    print("Initializing LLM Flask Server...")
    
    # Get machine IP and port
    host_ip = get_machine_ip()
    port = 5000
    
    # Print detailed host information
    print_host_details(host_ip, port)
    
    # Initialize Qwen instance at startup
    if initialize_qwen():
        print("\nStarting Flask-SocketIO server...")
        print(f"Access the server at: http://{host_ip}:{port}")
        print("Press Ctrl+C to stop the server")
        print("-" * 60)
        
        # Run the server
        socketio.run(
            app, 
            debug=True, 
            host=host_ip,  # Use machine IP instead of localhost
            port=port,
            use_reloader=False  # Disable reloader to prevent double initialization
        )
    else:
        print("Failed to initialize Qwen instance. Server not started.")
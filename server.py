from flask import Flask, render_template_string, send_from_directory, abort, jsonify, request
import os
import uuid
import threading
from pathlib import Path
from lustify_xwork import ImageGenerator

app = Flask(__name__)
#
#
#   https://claude.ai/chat/6f0a0290-9b3b-4c54-b719-fba287a52802
#

# Configuration
IMAGE_FOLDER = 'xserver'
FAVORITES_FILE = f'./{IMAGE_FOLDER}/fav.txt'
ALLOWED_EXTENSIONS = {'.png'}

""" SERVER STRUCTURE
xserver/
├── Model_photos/
│   ├── beach.png          ← Shows on 
│   ├── sunset.png         ← Shows on 
│   └── shoot_photos/  ← Link appears in header
│       ├── group1.png     ← Shows on 
│       └── group2.png     ← Shows on 
fav.txt                    ← Favorites file
"""

# Ensure the main image folder exists
os.makedirs(IMAGE_FOLDER, exist_ok=True)
# Ensure demo folders exist for generated images
os.makedirs(f"{IMAGE_FOLDER}/demo/demo_shoot", exist_ok=True)

# Global dict to track image generation status
generation_status = {}

def load_favorites():
    """Load favorites from fav.txt file."""
    try:
        if os.path.exists(FAVORITES_FILE):
            with open(FAVORITES_FILE, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())
        return set()
    except Exception:
        return set()

def save_favorites(favorites):
    """Save favorites to fav.txt file."""
    try:
        with open(FAVORITES_FILE, 'w', encoding='utf-8') as f:
            for fav in sorted(favorites):
                f.write(f"{fav}\n")
        return True
    except Exception:
        return False

def add_favorite(image_path):
    """Add an image to favorites."""
    favorites = load_favorites()
    favorites.add(image_path)
    return save_favorites(favorites)

def remove_favorite(image_path):
    """Remove an image from favorites."""
    favorites = load_favorites()
    favorites.discard(image_path)
    return save_favorites(favorites)

def is_favorite(image_path):
    """Check if an image is in favorites."""
    favorites = load_favorites()
    return image_path in favorites

def generate_image_thread(prompt, job_id):
    """Background thread function to generate image."""
    try:
        generation_status[job_id] = {
            'status': 'generating',
            'message': 'Generating image...',
            'progress': 0
        }
        
        generator = ImageGenerator()
        output_path = f"./xserver/demo/demo_shoot/demo_{str(uuid.uuid4())}.png"
        
        # Update progress
        generation_status[job_id]['progress'] = 50
        generation_status[job_id]['message'] = 'Processing prompt...'
        
        # Generate the image
        image = generator.text_to_image(
            prompt=prompt,
            output_path=output_path
        )
        
        # Success
        generation_status[job_id] = {
            'status': 'completed',
            'message': 'Image generated successfully!',
            'progress': 100,
            'image_path': output_path.replace('./xserver/', ''),
            'filename': os.path.basename(output_path)
        }
        
    except Exception as e:
        generation_status[job_id] = {
            'status': 'error',
            'message': f'Error generating image: {str(e)}',
            'progress': 0
        }

def get_subfolders():
    """Get all subdirectories in the main image folder."""
    try:
        main_path = Path(IMAGE_FOLDER)
        if not main_path.exists():
            return []
        
        folders = []
        for item in main_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                folders.append(item.name)
        
        return sorted(folders)
    except Exception:
        return []

def get_images_in_folder(folder_path):
    """Get all PNG images in a specific folder path (can include subfolders)."""
    try:
        if '/' in folder_path:
            # Handle subfolder case
            full_path = Path(IMAGE_FOLDER) / folder_path
        else:
            # Handle main folder case
            full_path = Path(IMAGE_FOLDER) / folder_path
            
        if not full_path.exists() or not full_path.is_dir():
            return []
        
        images = []
        for item in full_path.iterdir():
            if item.is_file() and item.suffix.lower() in ALLOWED_EXTENSIONS:
                images.append(item.name)
        
        return sorted(images)
    except Exception:
        return []

def get_subfolders_in_folder(folder_name):
    """Get all subdirectories in a specific folder."""
    try:
        folder_path = Path(IMAGE_FOLDER) / folder_name
        if not folder_path.exists() or not folder_path.is_dir():
            return []
        
        subfolders = []
        for item in folder_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                subfolders.append(item.name)
        
        return sorted(subfolders)
    except Exception:
        return []

@app.route('/')
def index():
    """Main page showing all subfolders."""
    folders = get_subfolders()
    return render_template_string(INDEX_TEMPLATE, 
                                folders=folders, 
                                image_folder=IMAGE_FOLDER)

@app.route('/image_prompt')
def image_prompt():
    """Image generation page."""
    return render_template_string(IMAGE_PROMPT_TEMPLATE)

@app.route('/api/generate', methods=['POST'])
def generate_image():
    """API endpoint to start image generation."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        print(data)
        # Expected fields with their default values
        expected_fields = {
            'style': 'photograph, photo of',
            'subject': 'a woman',
            'skin': 'tan skin',
            'hair': 'short face-framing blond hair with bangs',
            'face': 'high cheekbones',
            'eyes': 'brown eyes',
            'attribute': 'long eyelashes',
            'lips': 'full lips',
            'chest': 'wide',
            'pose':'sitting in a chair',
            'action':'legs crossed, barefoot',
            'framing':'sedutive stare at viewer',
            'clothes':'wearing a gold bikini',
            'lighting':'soft lighting from window'
        }
        
        # Extract and validate fields
        fields = {}
        for field_name, default_value in expected_fields.items():
            field_value = data.get(field_name, '').strip()
            if field_value:  # Only include non-empty fields
                fields[field_name] = field_value
        
        # Check if at least subject is provided
        if 'subject' not in fields:
            return jsonify({
                'success': False,
                'error': 'Subject field is required'
            }), 400
        
        # Concatenate all fields into a single prompt
        # Order matters for better prompt structure
        prompt_parts = []
        field_order = ['style', 'subject', 'skin', 'hair', 'face', 'eyes', 'attribute', 'lips', 'chest','pose','action',
                       'framing','clothes','lighting']
        
        for field_name in field_order:
            if field_name in fields:
                prompt_parts.append(fields[field_name])
        
        # Join with commas and spaces for natural language flow
        prompt = ', '.join(prompt_parts)
        
        # Clean up the prompt (remove extra commas, spaces)
        prompt = ', '.join([part.strip() for part in prompt.split(',') if part.strip()])
        
        if not prompt:
            return jsonify({
                'success': False,
                'error': 'Generated prompt is empty'
            }), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Log the generated prompt for debugging
        print(f"Generated prompt for job {job_id}: {prompt}")
        
        # Start generation in background thread
        thread = threading.Thread(
            target=generate_image_thread,
            args=(prompt, job_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Image generation started',
            'prompt': prompt  # Return the generated prompt for debugging
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }), 500

@app.route('/api/status/<job_id>')
def get_generation_status(job_id):
    """API endpoint to check image generation status."""
    if job_id not in generation_status:
        return jsonify({
            'success': False,
            'error': 'Job ID not found'
        }), 404
    
    status = generation_status[job_id]
    return jsonify({
        'success': True,
        **status
    })

@app.route('/folder/<folder_name>')
@app.route('/folder/<folder_name>/<subfolder_name>')
def view_folder(folder_name, subfolder_name=None):
    """Display all PNG images in a specific folder or subfolder."""
    # Security check - prevent directory traversal
    if '..' in folder_name or '/' in folder_name or '\\' in folder_name:
        abort(404)
    
    if subfolder_name:
        if '..' in subfolder_name or '/' in subfolder_name or '\\' in subfolder_name:
            abort(404)
        
        # Handle subfolder
        full_path = Path(IMAGE_FOLDER) / folder_name / subfolder_name
        display_name = f"{folder_name} / {subfolder_name}"
        folder_path_for_images = f"{folder_name}/{subfolder_name}"
        
        # Get parent folder link
        parent_folder = folder_name
        subfolders = []  # Subfolders don't have their own subfolders for now
    else:
        # Handle main folder
        full_path = Path(IMAGE_FOLDER) / folder_name
        display_name = folder_name
        folder_path_for_images = folder_name
        parent_folder = None
        
        # Get subfolders in this folder
        subfolders = get_subfolders_in_folder(folder_name)
    
    # Check if folder exists
    if not full_path.exists() or not full_path.is_dir():
        abort(404)
    
    # Get images in the current folder/subfolder
    if subfolder_name:
        images = get_images_in_folder(f"{folder_name}/{subfolder_name}")
    else:
        images = get_images_in_folder(folder_name)
    
    # Get favorites status for all images in this folder
    favorites = load_favorites()
    image_favorites = {}
    for image in images:
        image_path = f"{folder_path_for_images}/{image}"
        image_favorites[image_path] = image_path in favorites
    
    return render_template_string(FOLDER_TEMPLATE,
                                folder_name=display_name,
                                folder_path=folder_path_for_images,
                                images=images,
                                image_count=len(images),
                                subfolders=subfolders,
                                parent_folder=parent_folder,
                                favorites=image_favorites)

@app.route('/image/<path:folder_path>/<filename>')
def serve_image(folder_path, filename):
    """Serve individual image files from folder or subfolder."""
    full_folder_path = os.path.join(IMAGE_FOLDER, folder_path)
    response = send_from_directory(full_folder_path, filename)
    # Prevent browser cache if file is updated
    response.headers['Cache-Control'] = 'no-cache, must-revalidate'
    # Add Last-Modified header based on file mtime
    file_path = os.path.join(full_folder_path, filename)
    if os.path.exists(file_path):
        mtime = os.path.getmtime(file_path)
        from datetime import datetime
        last_modified = datetime.utcfromtimestamp(mtime).strftime('%a, %d %b %Y %H:%M:%S GMT')
        response.headers['Last-Modified'] = last_modified
    return response

@app.route('/api/favorite', methods=['POST'])
def handle_favorite():
    """API endpoint to add or remove favorites."""
    try:
        data = request.get_json()
        
        action = data['action']
        image_path = data['image_path']
        
        # Handle the favorite action
        if action == 'add':
            success = add_favorite(image_path)
        elif action == 'remove':
            success = remove_favorite(image_path)
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid action. Use "add" or "remove"'
            }), 400
        
        if success:
            return jsonify({
                'success': True,
                'action': action,
                'image_path': image_path
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to update favorites file'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/favorites')
def view_favorites():
    """Display all favorited images."""
    favorites = load_favorites()
    
    if not favorites:
        return render_template_string(FOLDER_TEMPLATE,
                                    folder_name="Favorites",
                                    folder_path="favorites",
                                    images=[],
                                    image_count=0,
                                    subfolders=[],
                                    parent_folder=None,
                                    favorites={})
    
    # Parse favorite paths and organize by folder
    favorite_images = []
    image_favorites = {}
    
    for fav_path in favorites:
        # Verify the image still exists
        full_path = os.path.join(IMAGE_FOLDER, fav_path)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            # Extract filename from path
            filename = os.path.basename(fav_path)
            favorite_images.append({
                'name': filename,
                'path': fav_path,
                'folder': os.path.dirname(fav_path)
            })
            image_favorites[fav_path] = True
    
    return render_template_string(FAVORITES_TEMPLATE,
                                folder_name="Favorites",
                                folder_path="favorites",
                                favorite_images=favorite_images,
                                image_count=len(favorite_images),
                                favorites=image_favorites)

@app.route('/api/delete', methods=['POST'])
def handle_delete():
    """API endpoint to delete favorited images from disk and remove from favorites."""
    try:
        data = request.get_json()
        
        if not data or 'image_path' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: image_path'
            }), 400
        
        image_path = data['image_path']
        
        # Security check - prevent directory traversal
        if '..' in image_path or image_path.startswith('/') or '\\' in image_path:
            return jsonify({
                'success': False,
                'error': 'Invalid image path'
            }), 400
        
        # Verify the image exists in favorites
        favorites = load_favorites()
        if image_path not in favorites:
            return jsonify({
                'success': False,
                'error': 'Image is not in favorites'
            }), 404
        
        # Get the full file path
        full_image_path = os.path.join(IMAGE_FOLDER, image_path)
        
        # Check if the file exists on disk
        if not os.path.exists(full_image_path) or not os.path.isfile(full_image_path):
            # File doesn't exist on disk, but remove it from favorites anyway
            remove_favorite(image_path)
            return jsonify({
                'success': True,
                'message': 'File not found on disk but removed from favorites',
                'image_path': image_path
            })
        
        try:
            # Delete the file from disk
            os.remove(full_image_path)
            
            # Remove from favorites
            remove_favorite(image_path)
            
            return jsonify({
                'success': True,
                'message': 'File deleted successfully and removed from favorites',
                'image_path': image_path
            })
            
        except PermissionError:
            return jsonify({
                'success': False,
                'error': 'Permission denied - cannot delete file'
            }), 403
            
        except OSError as e:
            return jsonify({
                'success': False,
                'error': f'Failed to delete file: {str(e)}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }), 500

@app.route('/api/delete_all_favorites', methods=['POST'])
def handle_delete_all_favorites():
    """API endpoint to delete ALL favorited images from disk and clear the favorites list."""
    try:
        favorites = load_favorites()
        
        if not favorites:
            return jsonify({
                'success': True,
                'message': 'No favorites to delete',
                'deleted_count': 0,
                'errors': []
            })
        
        deleted_count = 0
        errors = []
        
        # Delete each favorited file
        for image_path in list(favorites):  # Convert to list to avoid modification during iteration
            full_image_path = os.path.join(IMAGE_FOLDER, image_path)
            
            try:
                if os.path.exists(full_image_path) and os.path.isfile(full_image_path):
                    os.remove(full_image_path)
                    deleted_count += 1
                else:
                    errors.append(f"File not found: {image_path}")
                    
            except PermissionError:
                errors.append(f"Permission denied: {image_path}")
            except OSError as e:
                errors.append(f"Failed to delete {image_path}: {str(e)}")
        
        # Clear the favorites list
        save_favorites(set())
        
        return jsonify({
            'success': True,
            'message': f'Deleted {deleted_count} files and cleared favorites',
            'deleted_count': deleted_count,
            'total_favorites': len(favorites),
            'errors': errors
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }), 500
   
# HTML Templates
INDEX_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Server</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
            line-height: 1.6;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #333;
            margin-bottom: 30px;
            text-align: center;
            font-weight: 600;
        }
        
        .action-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        
        .action-button {
            display: inline-block;
            color: white;
            text-decoration: none;
            padding: 12px 24px;
            border-radius: 25px;
            font-weight: 500;
            transition: transform 0.2s ease;
        }
        
        .action-button:hover {
            transform: translateY(-2px);
            color: white;
            text-decoration: none;
        }
        
        .generate-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .favorites-button {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        }
        
        .folder-list {
            display: grid;
            gap: 15px;
        }
        
        .folder-item {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s ease;
        }
        
        .folder-item:hover {
            transform: translateY(-2px);
        }
        
        .folder-link {
            display: block;
            color: white;
            text-decoration: none;
            padding: 20px;
            font-size: 18px;
            font-weight: 500;
        }
        
        .folder-link:hover {
            color: white;
            text-decoration: none;
        }
        
        .no-folders {
            text-align: center;
            color: #666;
            padding: 40px;
            font-size: 16px;
        }
        
        @media (max-width: 600px) {
            body { padding: 10px; }
            .container { padding: 15px; }
            h1 { font-size: 24px; }
            .folder-link { padding: 15px; font-size: 16px; }
            .action-buttons { flex-direction: column; align-items: center; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📁 Image Server</h1>

        <!-- Action Buttons -->
        <div class="action-buttons">
            <a href="/image_prompt" class="action-button generate-button">
                🎨 Generate New Image
            </a>
            <a href="/favorites" class="action-button favorites-button">
                ❤️ View Favorites
            </a>
        </div>
        
        {% if folders %}
            <div class="folder-list">
                {% for folder in folders %}
                    <div class="folder-item">
                        <a href="/folder/{{ folder }}" class="folder-link">
                            📂 {{ folder }}
                        </a>
                    </div>
                {% endfor %}
            </div>
        {% else %}
            <div class="no-folders">
                No image folders found. Add some subfolders to '{{ image_folder }}' with .png files.
            </div>
        {% endif %}
    </div>
</body>
</html>
'''

IMAGE_PROMPT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generate Image - Image Server</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
            line-height: 1.6;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #333;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .back-link {
            display: inline-block;
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
            margin-bottom: 20px;
        }
        
        .back-link:hover {
            text-decoration: underline;
        }
        
        .generate-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 30px;
            color: white;
        }
        
        .generate-section h2 {
            margin-bottom: 20px;
            font-size: 24px;
            font-weight: 600;
            text-align: center;
        }
        
        .form-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .form-label {
            font-weight: 500;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            opacity: 0.9;
        }
        
        .form-input {
            padding: 12px 16px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            outline: none;
            font-family: inherit;
            background: rgba(255,255,255,0.95);
            color: #333;
        }
        
        .form-input:focus {
            background: white;
            box-shadow: 0 0 0 3px rgba(255,255,255,0.3);
        }
        
        .button-group {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 10px;
        }
        
        .generate-btn, .clear-btn {
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            min-width: 120px;
        }
        
        .generate-btn {
            background: rgba(255,255,255,0.9);
            color: #667eea;
        }
        
        .generate-btn:hover {
            background: white;
            transform: translateY(-2px);
        }
        
        .generate-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .clear-btn {
            background: rgba(255,255,255,0.2);
            color: white;
            border: 2px solid rgba(255,255,255,0.3);
        }
        
        .clear-btn:hover {
            background: rgba(255,255,255,0.3);
            border-color: rgba(255,255,255,0.5);
        }
        
        .status-message {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
            display: none;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.2);
            border-radius: 4px;
            margin-top: 15px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: white;
            width: 0%;
            transition: width 0.3s ease;
            border-radius: 4px;
        }
        
        .tips-section {
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 12px;
        }
        
        .tips-section h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 18px;
        }
        
        .tips-list {
            color: #666;
            line-height: 1.8;
        }
        
        .tips-list li {
            margin-bottom: 8px;
        }
        
        @media (max-width: 768px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 600px) {
            body { padding: 10px; }
            .container { padding: 20px; }
            .generate-section { padding: 20px; }
            .button-group { flex-direction: column; align-items: center; }
            .generate-btn, .clear-btn { width: 100%; max-width: 200px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-link">← Back to Image Folders</a>
        
        <div class="header">
            <h1>🎨 Generate New Image</h1>
            <p style="color: #666;">Create stunning images using detailed attributes</p>
        </div>

        <div class="generate-section">
            <h2>🖼️ Customize Your Image</h2>
            <div class="form-container">
                <div class="form-grid">
                    <div class="form-group">
                        <label class="form-label" for="style">Style</label>
                        <input 
                            type="text" 
                            class="form-input" 
                            id="style" 
                            value="photograph, photo of"
                            placeholder="e.g., photograph, digital art, painting"
                        >
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" for="subject">Subject</label>
                        <input 
                            type="text" 
                            class="form-input" 
                            id="subject" 
                            value="a woman"
                            placeholder="e.g., a woman, a man, a person"
                        >
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" for="skin">Skin</label>
                        <input 
                            type="text" 
                            class="form-input" 
                            id="skin" 
                            value="tan skin"
                            placeholder="e.g., tan skin, pale skin, dark skin"
                        >
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" for="hair">Hair</label>
                        <input 
                            type="text" 
                            class="form-input" 
                            id="hair" 
                            value="short face-framing blond hair with bangs"
                            placeholder="e.g., flowing brown hair, short layered hair, bixi cut, pixi"
                        >
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" for="face">Face</label>
                        <input 
                            type="text" 
                            class="form-input" 
                            id="face" 
                            value="high cheekbones"
                            placeholder="e.g., high cheekbones, soft features, angular features"
                        >
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" for="eyes">Eyes</label>
                        <input 
                            type="text" 
                            class="form-input" 
                            id="eyes" 
                            value="light blue eyes"
                            placeholder="e.g., blue eyes, green eyes, brown eyes"
                        >
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" for="attribute">Attribute</label>
                        <input 
                            type="text" 
                            class="form-input" 
                            id="attribute" 
                            value="long eyelashes"
                            placeholder="e.g., long eyelashes, freckles, dimples"
                        >
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" for="lips">Lips</label>
                        <input 
                            type="text" 
                            class="form-input" 
                            id="lips" 
                            value="full lips"
                            placeholder="e.g., full lips, thin lips, glossy lips"
                        >
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" for="chest">Chest</label>
                        <input 
                            type="text" 
                            class="form-input" 
                            id="chest" 
                            value="small tits"
                            placeholder="e.g., natural breasts, large tits"
                        >
                    </div>

                    <div class="form-group">
                        <label class="form-label" for="pose">Pose</label>
                        <input 
                            type="text" 
                            class="form-input" 
                            id="pose" 
                            value="sitting on a bed"
                            placeholder="e.g., laying on a blankiet, kneeling down"
                        >
                    </div>

                    <div class="form-group">
                        <label class="form-label" for="action">Action</label>
                        <input 
                            type="text" 
                            class="form-input" 
                            id="action" 
                            value="one leg crossed over the other, barefoot"
                            placeholder="e.g., walking, jumping"
                        >
                    </div>

                    <div class="form-group">
                        <label class="form-label" for="framing">Framing</label>
                        <input 
                            type="text" 
                            class="form-input" 
                            id="framing" 
                            value="seductive smile at viewer"
                            placeholder="e.g., looking up at viewer, looking back over shoulder"
                        >
                    </div>

                    <div class="form-group">
                        <label class="form-label" for="clothes">Clothes</label>
                        <input 
                            type="text" 
                            class="form-input" 
                            id="clothes" 
                            value="wearing a gold bikini"
                            placeholder="e.g., nude, underwear"
                        >
                    </div>

                    <div class="form-group">
                        <label class="form-label" for="lighting">Lighting</label>
                        <input 
                            type="text" 
                            class="form-input" 
                            id="lighting" 
                            value="soft light, 8k image"
                            placeholder="e.g., sunray through window, daylight"
                        >
                    </div>

                </div>
                
                <div class="button-group">
                    <button class="generate-btn" id="generateBtn" onclick="generateImage()">
                        Generate Image
                    </button>
                    <button class="clear-btn" onclick="clearForm()">
                        Reset to Defaults
                    </button>
                </div>
            </div>
            
            <div class="status-message" id="statusMessage">
                <div id="statusText"></div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
            </div>
        </div>

        <div class="tips-section">
            <h3>💡 Tips for Better Results</h3>
            <ul class="tips-list">
                <li><strong>Style:</strong> Try "photorealistic", "digital art", "oil painting", "watercolor", "cinematic"</li>
                <li><strong>Be specific:</strong> Use detailed descriptions for each attribute to get more accurate results</li>
                <li><strong>Combine attributes:</strong> The fields will be combined into a single prompt automatically</li>
                <li><strong>Experiment:</strong> Try different combinations of features to create unique looks</li>
                <li><strong>Leave empty:</strong> If you don't want a specific attribute, just leave that field empty</li>
            </ul>
        </div>
    </div>

    <script>
        let currentJobId = null;
        let statusInterval = null;

        function generateImage() {
            const generateBtn = document.getElementById('generateBtn');
            const statusMessage = document.getElementById('statusMessage');
            const statusText = document.getElementById('statusText');
            const progressFill = document.getElementById('progressFill');

            // Collect all form values
            const formData = {
                style: document.getElementById('style').value.trim(),
                subject: document.getElementById('subject').value.trim(),
                skin: document.getElementById('skin').value.trim(),
                hair: document.getElementById('hair').value.trim(),
                face: document.getElementById('face').value.trim(),
                eyes: document.getElementById('eyes').value.trim(),
                attribute: document.getElementById('attribute').value.trim(),
                lips: document.getElementById('lips').value.trim(),
                chest: document.getElementById('chest').value.trim(),
                pose: document.getElementById('pose').value.trim(),
                action: document.getElementById('action').value.trim(),
                framing: document.getElementById('framing').value.trim(),
                clothes: document.getElementById('clothes').value.trim(),
                lighting: document.getElementById('lighting').value.trim(),
            };

            // Check if at least subject is filled
            if (!formData.subject) {
                alert('Please enter at least a subject');
                return;
            }

            // Disable button and show status
            generateBtn.disabled = true;
            generateBtn.textContent = 'Generating...';
            statusMessage.style.display = 'block';
            statusText.textContent = 'Starting image generation...';
            progressFill.style.width = '0%';

            // Start generation
            fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    currentJobId = data.job_id;
                    checkStatus();
                } else {
                    throw new Error(data.error);
                }
            })
            .catch(error => {
                statusText.textContent = 'Error: ' + error.message;
                resetForm();
            });
        }

        function checkStatus() {
            if (!currentJobId) return;

            statusInterval = setInterval(() => {
                fetch(`/api/status/${currentJobId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const statusText = document.getElementById('statusText');
                        const progressFill = document.getElementById('progressFill');

                        statusText.textContent = data.message;
                        progressFill.style.width = data.progress + '%';

                        if (data.status === 'completed') {
                            clearInterval(statusInterval);
                            statusText.innerHTML = `✅ ${data.message}<br><a href="/folder/demo/demo_shoot" style="color: white; text-decoration: underline; font-weight: bold;">→ View Generated Image</a>`;
                            setTimeout(resetForm, 5000);
                        } else if (data.status === 'error') {
                            clearInterval(statusInterval);
                            statusText.textContent = '❌ ' + data.message;
                            setTimeout(resetForm, 3000);
                        }
                    }
                })
                .catch(error => {
                    clearInterval(statusInterval);
                    document.getElementById('statusText').textContent = 'Error checking status: ' + error.message;
                    setTimeout(resetForm, 3000);
                });
            }, 6000);
        }

        function resetForm() {
            const generateBtn = document.getElementById('generateBtn');
            const statusMessage = document.getElementById('statusMessage');

            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate Image';
            statusMessage.style.display = 'none';
            currentJobId = null;
            
            if (statusInterval) {
                clearInterval(statusInterval);
                statusInterval = null;
            }
        }

        function clearForm() {
            // Reset all fields to their default values
            document.getElementById('style').value = 'photograph, photo of';
            document.getElementById('subject').value = 'a woman';
            document.getElementById('skin').value = 'tan skin';
            document.getElementById('hair').value = 'short face-framing blond hair with bangs';
            document.getElementById('face').value = 'high cheekbones';
            document.getElementById('eyes').value = 'brown eyes';
            document.getElementById('attribute').value = 'long eyelashes';
            document.getElementById('lips').value = 'full lips';
            document.getElementById('chest').value = 'small tits';
            document.getElementById('pose').value = 'sitting in a white arm chair ';
            document.getElementById('action').value = 'crosslegged, barefoot';
            document.getElementById('framing').value = 'staring seductivly at viewer';
            document.getElementById('clothes').value = 'gold bikini';
            document.getElementById('lighting').value = 'sunlight';
            
            // Focus on the first field
            document.getElementById('style').focus();
        }

        // Allow Ctrl+Enter to submit from any field
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                generateImage();
            }
        });

        // Focus on style input when page loads
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('style').focus();
        });
    </script>
</body>
</html>
'''

FOLDER_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ folder_name }} - Images</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
        }
        
        .header {
            max-width: 1200px;
            margin: 0 auto 30px;
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .back-link {
            display: inline-block;
            color: #667eea;
            text-decoration: none;
            margin-bottom: 15px;
            font-weight: 500;
        }
        
        .back-link:hover {
            text-decoration: underline;
        }
        
        h1 {
            color: #333;
            font-weight: 600;
        }
        
        .image-count {
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }
        
        .subfolders {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }
        
        .subfolders h3 {
            color: #333;
            font-size: 16px;
            margin-bottom: 15px;
            font-weight: 600;
        }
        
        .subfolder-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .subfolder-link {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
            transition: transform 0.2s ease;
        }
        
        .subfolder-link:hover {
            color: white;
            text-decoration: none;
            transform: translateY(-1px);
        }
        
        .image-grid {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
        }
        
        .image-item {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
            position: relative;
        }
        
        .image-item:hover {
            transform: translateY(-2px);
        }
        
        .image-wrapper {
            width: 100%;
            height: 200px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f8f9fa;
            position: relative;
        }
        
        .image-wrapper img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            transition: transform 0.2s ease;
            cursor: pointer;
        }
        
        .image-item:hover img {
            transform: scale(1.05);
        }
        
        .action-buttons {
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            gap: 5px;
            z-index: 10;
        }
        
        .action-btn {
            background: rgba(255, 255, 255, 0.9);
            border: none;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 18px;
            transition: all 0.2s ease;
            backdrop-filter: blur(5px);
        }
        
        .action-btn:hover {
            background: rgba(255, 255, 255, 1);
            transform: scale(1.1);
        }
        
        .favorite-btn.favorited {
            background: rgba(255, 69, 100, 0.9);
            color: white;
        }
        
        .favorite-btn.favorited:hover {
            background: rgba(255, 69, 100, 1);
        }
        
        .delete-btn {
            background: rgba(255, 99, 71, 0.9);
            color: white;
        }
        
        .delete-btn:hover {
            background: rgba(255, 99, 71, 1);
        }
        
        .delete-btn:disabled {
            background: rgba(128, 128, 128, 0.5);
            cursor: not-allowed;
            transform: none;
        }
        
        .action-btn.loading {
            opacity: 0.6;
            pointer-events: none;
        }
        
        /* Modal Styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.95);
            backdrop-filter: blur(5px);
        }
        
        .modal-content {
            position: relative;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .modal-image {
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
            user-select: none;
        }
        
        .close {
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            z-index: 1001;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            line-height: 1;
        }
        
        .close:hover {
            background: rgba(0, 0, 0, 0.8);
        }
        
        .nav-button {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(0, 0, 0, 0.5);
            color: white;
            border: none;
            font-size: 24px;
            padding: 15px 20px;
            cursor: pointer;
            border-radius: 8px;
            z-index: 1001;
            user-select: none;
        }
        
        .nav-button:hover {
            background: rgba(0, 0, 0, 0.8);
        }
        
        .nav-button:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }
        
        .prev {
            left: 20px;
        }
        
        .next {
            right: 20px;
        }
        
        .image-counter {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            background: rgba(0, 0, 0, 0.5);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            z-index: 1001;
        }
        
        /* Confirmation Dialog */
        .confirm-dialog {
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(2px);
        }
        
        .confirm-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            max-width: 400px;
            width: 90%;
            text-align: center;
        }
        
        .confirm-content h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 18px;
        }
        
        .confirm-content p {
            color: #666;
            margin-bottom: 25px;
            line-height: 1.5;
        }
        
        .confirm-buttons {
            display: flex;
            gap: 10px;
            justify-content: center;
        }
        
        .confirm-btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            font-size: 14px;
            transition: all 0.2s ease;
        }
        
        .confirm-btn.delete {
            background: #ff6347;
            color: white;
        }
        
        .confirm-btn.delete:hover {
            background: #ff4500;
        }
        
        .confirm-btn.cancel {
            background: #f0f0f0;
            color: #333;
        }
        
        .confirm-btn.cancel:hover {
            background: #e0e0e0;
        }
        
        @media (max-width: 600px) {
            .close {
                top: 10px;
                right: 15px;
                font-size: 30px;
                width: 40px;
                height: 40px;
            }
            
            .nav-button {
                font-size: 20px;
                padding: 12px 15px;
            }
            
            .prev { left: 10px; }
            .next { right: 10px; }
            
            .modal-image {
                max-width: 95%;
                max-height: 85%;
            }
            
            .image-counter {
                bottom: 60px;
                font-size: 12px;
                padding: 6px 12px;
            }
            
            .action-btn {
                width: 32px;
                height: 32px;
                font-size: 16px;
            }
            
            .action-buttons {
                top: 8px;
                right: 8px;
                gap: 3px;
            }
        }
        
        .image-name {
            padding: 15px;
            font-size: 14px;
            color: #333;
            word-break: break-all;
            background: white;
        }
        
        .no-images {
            max-width: 1200px;
            margin: 0 auto;
            text-align: center;
            color: #666;
            padding: 40px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        @media (max-width: 600px) {
            body { padding: 10px; }
            .header { padding: 15px; }
            .image-grid {
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 15px;
            }
            .image-wrapper { height: 150px; }
            .image-name { padding: 10px; font-size: 12px; }
        }
        
        @media (max-width: 400px) {
            .image-grid {
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }
            .image-wrapper { height: 120px; }
        }
    </style>
</head>
<body>
    <div class="header">
        <a href="/" class="back-link">← Back to Folders</a>
        <h1>📂 {{ folder_name }}</h1>
        <div class="image-count">{{ image_count }} images</div>
        
        {% if subfolders %}
            <div class="subfolders">
                <h3>📁 Subfolders</h3>
                <div class="subfolder-list">
                    {% for subfolder in subfolders %}
                        <a href="/folder/{{ folder_name }}/{{ subfolder }}" class="subfolder-link">
                            {{ subfolder }}
                        </a>
                    {% endfor %}
                </div>
            </div>
        {% endif %}
    </div>
    
    {% if images %}
        <div class="image-grid">
            {% for image in images %}
                <div class="image-item">
                    <div class="image-wrapper">
                        <img src="/image/{{ folder_path }}/{{ image }}" 
                             alt="{{ image }}" 
                             loading="lazy"
                             onclick="openModal({{ loop.index0 }})">
                        <div class="action-buttons">
                            <button class="action-btn favorite-btn {% if favorites[folder_path + '/' + image] %}favorited{% endif %}" 
                                    onclick="toggleFavorite(event, '{{ folder_path }}/{{ image }}', this)"
                                    data-image-path="{{ folder_path }}/{{ image }}"
                                    title="Add to favorites">
                                {% if favorites[folder_path + '/' + image] %}❤️{% else %}🤍{% endif %}
                            </button>
                            {% if favorites[folder_path + '/' + image] %}
                            <button class="action-btn delete-btn" 
                                    onclick="confirmDelete(event, '{{ folder_path }}/{{ image }}', this)"
                                    data-image-path="{{ folder_path }}/{{ image }}"
                                    title="Delete image">
                                👎
                            </button>
                            {% endif %}
                        </div>
                    </div>
                    <div class="image-name">{{ image }}</div>
                </div>
            {% endfor %}
        </div>
        
        <!-- Modal -->
        <div id="imageModal" class="modal">
            <div class="modal-content">
                <span class="close" onclick="closeModal()">&times;</span>
                <button class="nav-button prev" onclick="changeImage(-1)">‹</button>
                <img class="modal-image" id="modalImage" src="" alt="">
                <button class="nav-button next" onclick="changeImage(1)">›</button>
                <div class="image-counter">
                    <span id="currentImageNum">1</span> / <span id="totalImages">{{ image_count }}</span>
                </div>
            </div>
        </div>
        
        <!-- Confirmation Dialog -->
        <div id="confirmDialog" class="confirm-dialog">
            <div class="confirm-content">
                <h3>Delete Image</h3>
                <p>Are you sure you want to permanently delete this image? This action cannot be undone.</p>
                <div class="confirm-buttons">
                    <button class="confirm-btn delete" onclick="executeDelete()">Delete</button>
                    <button class="confirm-btn cancel" onclick="cancelDelete()">Cancel</button>
                </div>
            </div>
        </div>
    {% else %}
        <div class="no-images">
            No .png images found in this folder.
        </div>
    {% endif %}
    
    <script>
        // Image data for carousel
        const images = [
            {% for image in images %}
            {
                src: "/image/{{ folder_path }}/{{ image }}",
                name: "{{ image }}",
                path: "{{ folder_path }}/{{ image }}"
            }{% if not loop.last %},{% endif %}
            {% endfor %}
        ];
        
        let currentImageIndex = 0;
        let pendingDeletePath = null;
        let pendingDeleteButton = null;
        const modal = document.getElementById('imageModal');
        const modalImage = document.getElementById('modalImage');
        const currentImageNum = document.getElementById('currentImageNum');
        const totalImages = document.getElementById('totalImages');
        const confirmDialog = document.getElementById('confirmDialog');
        
        function openModal(imageIndex) {
            currentImageIndex = imageIndex;
            showImage();
            modal.style.display = 'block';
            document.body.style.overflow = 'hidden'; // Prevent background scrolling
        }
        
        function closeModal() {
            modal.style.display = 'none';
            document.body.style.overflow = 'auto'; // Restore scrolling
        }
        
        function changeImage(direction) {
            currentImageIndex += direction;
            
            // Loop around if at ends
            if (currentImageIndex >= images.length) {
                currentImageIndex = 0;
            } else if (currentImageIndex < 0) {
                currentImageIndex = images.length - 1;
            }
            
            showImage();
        }
        
        function showImage() {
            if (images.length > 0) {
                modalImage.src = images[currentImageIndex].src;
                modalImage.alt = images[currentImageIndex].name;
                currentImageNum.textContent = currentImageIndex + 1;
                
                // Update navigation buttons
                const prevBtn = document.querySelector('.prev');
                const nextBtn = document.querySelector('.next');
                
                if (images.length <= 1) {
                    prevBtn.style.display = 'none';
                    nextBtn.style.display = 'none';
                } else {
                    prevBtn.style.display = 'block';
                    nextBtn.style.display = 'block';
                }
            }
        }
        
        // Favorites functionality
        async function toggleFavorite(event, imagePath, button) {
            // Prevent opening modal when clicking favorite button
            event.stopPropagation();
            
            // Add loading state
            button.classList.add('loading');
            
            const isFavorited = button.classList.contains('favorited');
            const action = isFavorited ? 'remove' : 'add';
            
            try {
                const response = await fetch('/api/favorite', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        action: action,
                        image_path: imagePath
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Update button state
                    if (action === 'add') {
                        button.classList.add('favorited');
                        button.textContent = '❤️';
                        // Show delete button
                        showDeleteButton(button, imagePath);
                    } else {
                        button.classList.remove('favorited');
                        button.textContent = '🤍';
                        // Hide delete button
                        hideDeleteButton(button);
                    }
                } else {
                    console.error('Failed to update favorite:', result.error);
                    alert('Failed to update favorite: ' + result.error);
                }
            } catch (error) {
                console.error('Error updating favorite:', error);
                alert('Error updating favorite. Please try again.');
            } finally {
                // Remove loading state
                button.classList.remove('loading');
            }
        }
        
        // Delete functionality
        function confirmDelete(event, imagePath, button) {
            // Prevent opening modal when clicking delete button
            event.stopPropagation();
            
            pendingDeletePath = imagePath;
            pendingDeleteButton = button;
            confirmDialog.style.display = 'block';
        }
        
        function cancelDelete() {
            confirmDialog.style.display = 'none';
            pendingDeletePath = null;
            pendingDeleteButton = null;
        }
        
        async function executeDelete() {
            if (!pendingDeletePath || !pendingDeleteButton) {
                cancelDelete();
                return;
            }
            
            // Hide dialog first
            confirmDialog.style.display = 'none';
            
            // Add loading state to button
            pendingDeleteButton.classList.add('loading');
            pendingDeleteButton.disabled = true;
            
            try {
                const response = await fetch('/api/delete', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image_path: pendingDeletePath
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Remove the entire image item from the page
                    const imageItem = pendingDeleteButton.closest('.image-item');
                    if (imageItem) {
                        imageItem.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                        imageItem.style.opacity = '0';
                        imageItem.style.transform = 'scale(0.8)';
                        
                        setTimeout(() => {
                            imageItem.remove();
                            // Update image count
                            updateImageCount();
                        }, 300);
                    }
                    
                    // Remove from images array for modal navigation
                    const imageIndex = images.findIndex(img => img.path === pendingDeletePath);
                    if (imageIndex !== -1) {
                        images.splice(imageIndex, 1);
                    }
                    
                } else {
                    console.error('Failed to delete image:', result.error);
                    alert('Failed to delete image: ' + result.error);
                    
                    // Remove loading state on error
                    pendingDeleteButton.classList.remove('loading');
                    pendingDeleteButton.disabled = false;
                }
            } catch (error) {
                console.error('Error deleting image:', error);
                alert('Error deleting image. Please try again.');
                
                // Remove loading state on error
                pendingDeleteButton.classList.remove('loading');
                pendingDeleteButton.disabled = false;
            } finally {
                // Clear pending delete
                pendingDeletePath = null;
                pendingDeleteButton = null;
            }
        }
        
        function showDeleteButton(favoriteButton, imagePath) {
            const actionButtons = favoriteButton.parentElement;
            const existingDeleteBtn = actionButtons.querySelector('.delete-btn');
            
            if (!existingDeleteBtn) {
                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'action-btn delete-btn';
                deleteBtn.onclick = (e) => confirmDelete(e, imagePath, deleteBtn);
                deleteBtn.setAttribute('data-image-path', imagePath);
                deleteBtn.setAttribute('title', 'Delete image');
                deleteBtn.textContent = '👎';
                actionButtons.appendChild(deleteBtn);
            }
        }
        
        function hideDeleteButton(favoriteButton) {
            const actionButtons = favoriteButton.parentElement;
            const deleteBtn = actionButtons.querySelector('.delete-btn');
            if (deleteBtn) {
                deleteBtn.remove();
            }
        }
        
        function updateImageCount() {
            const imageCountElement = document.querySelector('.image-count');
            const currentCount = document.querySelectorAll('.image-item').length;
            imageCountElement.textContent = currentCount + ' images';
            
            // Update modal counter
            totalImages.textContent = currentCount;
            
            // If no images left, show no images message
            if (currentCount === 0) {
                const imageGrid = document.querySelector('.image-grid');
                imageGrid.innerHTML = '<div class="no-images">No .png images found in this folder.</div>';
            }
        }
        
        // Keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (modal.style.display === 'block') {
                switch(e.key) {
                    case 'Escape':
                        closeModal();
                        break;
                    case 'ArrowLeft':
                        if (images.length > 1) changeImage(-1);
                        break;
                    case 'ArrowRight':
                        if (images.length > 1) changeImage(1);
                        break;
                }
            } else if (confirmDialog.style.display === 'block') {
                if (e.key === 'Escape') {
                    cancelDelete();
                }
            }
        });
        
        // Close modal when clicking outside the image
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                closeModal();
            }
        });
        
        // Close confirm dialog when clicking outside
        confirmDialog.addEventListener('click', function(e) {
            if (e.target === confirmDialog) {
                cancelDelete();
            }
        });
        
        // Touch/swipe support for mobile
        let touchStartX = 0;
        let touchEndX = 0;
        
        modal.addEventListener('touchstart', function(e) {
            touchStartX = e.changedTouches[0].screenX;
        });
        
        modal.addEventListener('touchend', function(e) {
            touchEndX = e.changedTouches[0].screenX;
            handleSwipe();
        });
        
        function handleSwipe() {
            const swipeThreshold = 50;
            const diff = touchStartX - touchEndX;
            
            if (Math.abs(diff) > swipeThreshold && images.length > 1) {
                if (diff > 0) {
                    // Swipe left - next image
                    changeImage(1);
                } else {
                    // Swipe right - previous image
                    changeImage(-1);
                }
            }
        }
    </script>
</body>
</html>
'''

FAVORITES_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Favorites - Images</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            padding: 10px;
        }
        
        .header {
            max-width: 1044px; /* 1024 + 20px padding */
            margin: 0 auto 15px;
            background: white;
            border-radius: 8px;
            padding: 15px 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .back-link {
            display: inline-block;
            color: #667eea;
            text-decoration: none;
            margin-bottom: 8px;
            font-weight: 500;
            font-size: 14px;
        }
        
        .back-link:hover {
            text-decoration: underline;
        }
        
        h1 {
            color: #333;
            font-weight: 600;
            font-size: 24px;
            margin-bottom: 2px;
        }
        
        .image-count {
            color: #666;
            font-size: 13px;
        }
        
        .image-grid {
            max-width: 1044px; /* 1024 + 20px padding */
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .image-item {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s ease;
            position: relative;
        }
        
        .image-item:hover {
            transform: translateY(-1px);
        }
        
        .image-wrapper {
            width: 100%;
            height: auto;
            position: relative;
            background: #fafafa;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 10px;
        }
        
        .image-wrapper img {
            width: 100%;
            height: auto;
            max-width: 1024px;
            display: block;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .favorite-btn {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255, 69, 100, 0.9);
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 20px;
            transition: all 0.2s ease;
            z-index: 10;
            backdrop-filter: blur(5px);
        }
        
        .favorite-btn:hover {
            background: rgba(255, 69, 100, 1);
            transform: scale(1.05);
        }
        
        .image-name {
            padding: 12px 20px;
            font-size: 14px;
            color: #333;
            word-break: break-word;
            background: white;
            line-height: 1.3;
            border-top: 1px solid #f0f0f0;
        }
        
        .no-images {
            max-width: 1044px;
            margin: 0 auto;
            text-align: center;
            color: #666;
            padding: 60px 40px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        /* Mobile responsiveness */
        @media (max-width: 1064px) {
            body { padding: 8px; }
            .header, .image-grid { max-width: calc(100vw - 16px); }
            .header { padding: 12px 16px; }
            .image-wrapper { padding: 8px; }
            .image-name { padding: 10px 16px; font-size: 13px; }
            .favorite-btn {
                width: 36px;
                height: 36px;
                font-size: 18px;
                top: 16px;
                right: 16px;
            }
        }
        
        @media (max-width: 600px) {
            body { padding: 5px; }
            .header { padding: 10px 12px; }
            h1 { font-size: 20px; }
            .image-wrapper { padding: 5px; }
            .image-name { padding: 8px 12px; font-size: 12px; }
            .favorite-btn {
                width: 32px;
                height: 32px;
                font-size: 16px;
                top: 12px;
                right: 12px;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <a href="/" class="back-link">← Back to Folders</a>
        <h1>❤️ Favorites</h1>
        <div class="image-count">{{ image_count }} favorite images</div>
    </div>
    
    {% if favorite_images %}
        <div class="image-grid">
            {% for image in favorite_images %}
                <div class="image-item">
                    <div class="image-wrapper">
                        <img src="/image/{{ image.path }}" 
                             alt="{{ image.name }}" 
                             loading="lazy">
                        <button class="favorite-btn favorited" 
                                onclick="toggleFavorite(event, '{{ image.path }}', this)"
                                data-image-path="{{ image.path }}">
                            ❤️
                        </button>
                    </div>
                    <div class="image-name">{{ image.name }}</div>
                </div>
            {% endfor %}
        </div>
    {% else %}
        <div class="no-images">
            No favorite images yet. Add some favorites by clicking the heart button on any image!
        </div>
    {% endif %}
    
    <script>
        // Favorites functionality
        async function toggleFavorite(event, imagePath, button) {
            event.stopPropagation();
            button.classList.add('loading');
            
            try {
                const response = await fetch('/api/favorite', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        action: 'remove',
                        image_path: imagePath
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Remove the image item from the favorites page
                    button.closest('.image-item').remove();
                    
                    // Update the count
                    const countElement = document.querySelector('.image-count');
                    const currentCount = parseInt(countElement.textContent.match(/\d+/)[0]);
                    const newCount = currentCount - 1;
                    countElement.textContent = `${newCount} favorite images`;
                    
                    // Show "no images" message if no favorites left
                    if (newCount === 0) {
                        location.reload();
                    }
                }
            } catch (error) {
                console.error('Error removing favorite:', error);
            } finally {
                button.classList.remove('loading');
            }
        }
    </script>
</body>
</html>
'''


if __name__ == '__main__':
    print(f"Starting Flask image server...")
    print(f"Image folder: {os.path.abspath(IMAGE_FOLDER)}")
    print(f"Favorites file: {os.path.abspath(FAVORITES_FILE)}")
    print(f"Server will be available at: http://localhost:5000")
    print(f"Add subfolders with .png files to '{IMAGE_FOLDER}' to see them appear!")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)


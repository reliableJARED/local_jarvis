from flask import Flask, render_template_string, send_from_directory, abort, jsonify, request
import os
from pathlib import Path

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
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📁 Image Folders</h1>

        <!-- Add this favorites button -->
        <div style="text-align: center; margin-bottom: 20px;">
            <a href="/favorites" style="display: inline-block; background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); color: white; text-decoration: none; padding: 12px 24px; border-radius: 25px; font-weight: 500; transition: transform 0.2s ease;">
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
        
        .favorite-btn {
            position: absolute;
            top: 10px;
            right: 10px;
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
            z-index: 10;
            backdrop-filter: blur(5px);
        }
        
        .favorite-btn:hover {
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
        
        .favorite-btn.loading {
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
            
            .favorite-btn {
                width: 32px;
                height: 32px;
                font-size: 16px;
                top: 8px;
                right: 8px;
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
                        <button class="favorite-btn {% if favorites[folder_path + '/' + image] %}favorited{% endif %}" 
                                onclick="toggleFavorite(event, '{{ folder_path }}/{{ image }}', this)"
                                data-image-path="{{ folder_path }}/{{ image }}">
                            {% if favorites[folder_path + '/' + image] %}❤️{% else %}🤍{% endif %}
                        </button>
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
        const modal = document.getElementById('imageModal');
        const modalImage = document.getElementById('modalImage');
        const currentImageNum = document.getElementById('currentImageNum');
        const totalImages = document.getElementById('totalImages');
        
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
                    } else {
                        button.classList.remove('favorited');
                        button.textContent = '🤍';
                    }
                } else {
                    console.error('Failed to update favorite:', result.error);
                    // You could add a toast notification here
                }
            } catch (error) {
                console.error('Error updating favorite:', error);
                // You could add a toast notification here
            } finally {
                // Remove loading state
                button.classList.remove('loading');
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
            }
        });
        
        // Close modal when clicking outside the image
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                closeModal();
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
        /* Use the same styles as FOLDER_TEMPLATE */
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
        
        .image-grid {
            max-width: 600px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }
        
        .image-item {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
            position: relative;
            display: flex;
            flex-direction: column;
            height: 280px;
        }
        
        .image-item:hover {
            transform: translateY(-2px);
        }
        
        .image-wrapper {
            width: 100%;
            flex: 1;
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
        
        .favorite-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255, 69, 100, 0.9);
            color: white;
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
            z-index: 10;
            backdrop-filter: blur(5px);
        }
        
        .favorite-btn:hover {
            background: rgba(255, 69, 100, 1);
            transform: scale(1.1);
        }
        
        .image-name {
            padding: 8px 12px 4px 12px;
            font-size: 12px;
            color: #333;
            word-break: break-all;
            background: white;
            flex-shrink: 0;
            line-height: 1.2;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .folder-path {
            padding: 0 12px 8px 12px;
            font-size: 10px;
            color: #666;
            font-style: italic;
            flex-shrink: 0;
            line-height: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
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
                max-width: 100%;
                gap: 15px;
            }
            .image-item { height: 200px; }
            .image-name { 
                padding: 6px 8px 3px 8px; 
                font-size: 11px; 
            }
            .folder-path { 
                padding: 0 8px 6px 8px; 
                font-size: 9px; 
            }
            .favorite-btn {
                width: 32px;
                height: 32px;
                font-size: 16px;
                top: 8px;
                right: 8px;
            }
        }
        
        @media (max-width: 400px) {
            .image-item { height: 180px; }
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
                    <div class="folder-path">{{ image.folder }}</div>
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
    """# Security checks
    if '..' in folder_path or '..' in filename:
        abort(404)
    
    # Check file extension
    if not filename.lower().endswith('.png'):
        abort(404)
    
    full_folder_path = os.path.join(IMAGE_FOLDER, folder_path)
    
    # Check if folder and file exist
    if not os.path.exists(full_folder_path) or not os.path.isdir(full_folder_path):
        abort(404)
    
    file_path = os.path.join(full_folder_path, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        abort(404)"""
    full_folder_path = os.path.join(IMAGE_FOLDER, folder_path)
    
    #Caching for faster loads
    response = send_from_directory(full_folder_path, filename)
    response.headers['Cache-Control'] = 'public, max-age=3600'
    return response

@app.route('/api/favorite', methods=['POST'])
def handle_favorite():
    """API endpoint to add or remove favorites."""
    try:
        data = request.get_json()
        
        """if not data or 'action' not in data or 'image_path' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: action and image_path'
            }), 400"""
        
        action = data['action']
        image_path = data['image_path']
        
        """# Security check - prevent directory traversal
        if '..' in image_path or image_path.startswith('/') or '\\' in image_path:
            return jsonify({
                'success': False,
                'error': 'Invalid image path'
            }), 400"""
        
        """# Verify the image exists
        full_image_path = os.path.join(IMAGE_FOLDER, image_path)
        if not os.path.exists(full_image_path) or not os.path.isfile(full_image_path):
            return jsonify({
                'success': False,
                'error': 'Image not found'
            }), 404"""
        
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

if __name__ == '__main__':
    print(f"Starting Flask image server...")
    print(f"Image folder: {os.path.abspath(IMAGE_FOLDER)}")
    print(f"Favorites file: {os.path.abspath(FAVORITES_FILE)}")
    print(f"Server will be available at: http://localhost:5000")
    print(f"Add subfolders with .png files to '{IMAGE_FOLDER}' to see them appear!")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)


from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import sqlite3
import os
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load YOLO model
model = YOLO('best.pt')

# Initialize database
def init_db():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  method TEXT,
                  total_eggs INTEGER,
                  total_cracks INTEGER,
                  image_path TEXT)''')
    conn.commit()
    conn.close()

init_db()

def save_detection(method, eggs, cracks, img_path):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''INSERT INTO detections (timestamp, method, total_eggs, total_cracks, image_path)
                 VALUES (?, ?, ?, ?, ?)''',
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), method, eggs, cracks, img_path))
    conn.commit()
    conn.close()

def process_image(img):
    results = model(img)
    
    eggs = 0
    cracks = 0
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if conf > 0.5:
                if cls == 0:  # egg
                    eggs += 1
                elif cls == 1:  # crack
                    cracks += 1
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                label = f"{'Egg' if cls == 0 else 'Crack'}: {conf:.2f}"
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 2)
    
    return img, eggs, cracks

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    img = cv2.imread(filepath)
    result_img, eggs, cracks = process_image(img)
    
    result_filename = f"result_{filename}"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, result_img)
    
    save_detection('upload', eggs, cracks, result_filename)
    
    return jsonify({
        'success': True,
        'eggs': eggs,
        'cracks': cracks,
        'image': f'/results/{result_filename}'
    })

@app.route('/webcam_frame', methods=['POST'])
def webcam_frame():
    data = request.json
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    result_img, eggs, cracks = process_image(img)
    
    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"webcam_{timestamp}.jpg"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, result_img)
    
    save_detection('webcam', eggs, cracks, result_filename)
    
    return jsonify({
        'success': True,
        'eggs': eggs,
        'cracks': cracks,
        'image': f'data:image/jpeg;base64,{img_base64}'
    })

@app.route('/history')
def history():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('SELECT * FROM detections ORDER BY id DESC LIMIT 50')
    rows = c.fetchall()
    conn.close()
    
    history_data = []
    for row in rows:
        history_data.append({
            'id': row[0],
            'timestamp': row[1],
            'method': row[2],
            'eggs': row[3],
            'cracks': row[4],
            'image': row[5]
        })
    
    return jsonify(history_data)

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
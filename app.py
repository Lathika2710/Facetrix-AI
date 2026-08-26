import os
import uuid
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import config
from database import (
    init_db, add_person, add_face_embedding,
    get_all_embeddings, get_all_persons_with_counts, delete_person
)
from services.face_detector import detect_faces
from services.face_embedding import extract_face_embedding
from services.face_matcher import match_face_embedding
from services.face_processor import draw_visualizations

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Initialize database on startup
init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home landing page with system metrics and pipeline breakdown."""
    persons = get_all_persons_with_counts()
    total_persons = len(persons)
    total_samples = sum(p['sample_count'] for p in persons)
    return render_template(
        'index.html',
        total_persons=total_persons,
        total_samples=total_samples,
        threshold=config.SIMILARITY_THRESHOLD
    )

@app.route('/webcam')
def webcam():
    """Real-time webcam stream face recognition page."""
    return render_template('webcam.html', threshold=config.SIMILARITY_THRESHOLD)

@app.route('/api/recognize_frame', methods=['POST'])
def api_recognize_frame():
    """
    API endpoint receiving base64 encoded JPEG frames from HTML5 WebRTC webcam.
    Performs real-time face detection, landmark alignment, ArcFace embedding extraction,
    and cosine similarity identity matching against SQLite database.
    """
    try:
        data = request.get_json(force=True)
        image_data = data.get('image', '')
        
        if not image_data or ',' not in image_data:
            return jsonify({'success': False, 'error': 'Invalid image payload'}), 400
            
        # Decode base64 string to BGR OpenCV image
        encoded_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(encoded_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None or img_bgr.size == 0:
            return jsonify({'success': False, 'error': 'Corrupted frame'}), 400
            
        h, w = img_bgr.shape[:2]
        
        # Detect faces in frame
        faces = detect_faces(img_bgr)
        if not faces:
            return jsonify({
                'success': True,
                'faces_detected': 0,
                'annotated_image': image_data,  # Return raw frame
                'results': []
            })
            
        stored_embeddings = get_all_embeddings()
        faces_visual_data = []
        recognition_results = []
        
        for face in faces:
            query_emb = extract_face_embedding(img_bgr, face_info=face)
            match_res = match_face_embedding(query_emb, stored_embeddings, threshold=config.SIMILARITY_THRESHOLD)
            
            face_viz = {
                'bbox': face['bbox'],
                'kps': face['kps'],
                'identity': match_res['name'],
                'similarity': match_res['similarity'],
                'status': match_res['status']
            }
            faces_visual_data.append(face_viz)
            recognition_results.append(match_res)
            
        # Draw sleek visual bounding boxes and 5 landmarks onto copy of frame
        annotated_bgr = draw_visualizations(img_bgr, faces_visual_data)
        
        # Encode annotated image back to Base64 JPEG for browser render
        _, buffer = cv2.imencode('.jpg', annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'faces_detected': len(faces),
            'annotated_image': annotated_base64,
            'dimensions': f"{w}x{h}",
            'aspect_ratio': f"{w/h:.2f}:1",
            'results': recognition_results
        })
        
    except Exception as e:
        app.logger.error("Webcam frame recognition error: %s", str(e), exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register a new person with multiple facial sample images or webcam snapshots."""
    if request.method == 'POST':
        person_name = request.form.get('name', '').strip()
        uploaded_files = request.files.getlist('images')
        webcam_snapshots = request.form.getlist('webcam_snapshots')
        
        if not person_name:
            flash('Please enter a valid person name or ID.', 'danger')
            return redirect(request.url)
            
        valid_files = [f for f in uploaded_files if f and f.filename and allowed_file(f.filename)]
        valid_snapshots = [s for s in webcam_snapshots if s and ',' in s]
        
        if not valid_files and not valid_snapshots:
            flash('Please upload at least one valid image file or capture a webcam snapshot.', 'warning')
            return redirect(request.url)
            
        person_id = add_person(person_name)
        success_count = 0
        failed_count = 0
        error_details = []
        
        # Process uploaded files
        for file in valid_files:
            try:
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img_bgr is None or img_bgr.size == 0:
                    failed_count += 1
                    error_details.append(f"{file.filename}: Corrupted or invalid image file.")
                    continue
                    
                faces = detect_faces(img_bgr)
                if not faces:
                    failed_count += 1
                    error_details.append(f"{file.filename}: No face detected.")
                    continue
                    
                emb = extract_face_embedding(img_bgr, face_info=faces[0])
                add_face_embedding(person_id, emb)
                success_count += 1
                
            except Exception as e:
                failed_count += 1
                error_details.append(f"{file.filename}: {str(e)}")
                
        # Process webcam snapshots
        for idx, b64_str in enumerate(valid_snapshots, 1):
            try:
                encoded_data = b64_str.split(',')[1]
                img_bytes = base64.b64decode(encoded_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if img_bgr is None or img_bgr.size == 0:
                    failed_count += 1
                    error_details.append(f"Webcam Snapshot #{idx}: Corrupted frame.")
                    continue
                    
                faces = detect_faces(img_bgr)
                if not faces:
                    failed_count += 1
                    error_details.append(f"Webcam Snapshot #{idx}: No face detected in snapshot.")
                    continue
                    
                emb = extract_face_embedding(img_bgr, face_info=faces[0])
                add_face_embedding(person_id, emb)
                success_count += 1
                
            except Exception as e:
                failed_count += 1
                error_details.append(f"Webcam Snapshot #{idx}: {str(e)}")
                
        if success_count > 0:
            flash(
                f"Successfully registered '{person_name}' with {success_count} face embedding sample(s)! "
                f"({failed_count} failed)",
                'success'
            )
        else:
            flash(f"Failed to register '{person_name}'. No valid faces could be processed.", 'danger')
            
        return render_template(
            'register.html',
            registration_result={
                'name': person_name,
                'success_count': success_count,
                'failed_count': failed_count,
                'errors': error_details
            }
        )
        
    return render_template('register.html')

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    """Recognize identity from a query image of arbitrary resolution and aspect ratio."""
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image file selected.', 'warning')
            return redirect(request.url)
            
        file = request.files['image']
        if not file or not file.filename or not allowed_file(file.filename):
            flash('Invalid image file. Allowed types: PNG, JPG, JPEG, WEBP.', 'danger')
            return redirect(request.url)
            
        try:
            # Read query image
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img_bgr is None or img_bgr.size == 0:
                flash('Uploaded file is corrupted or unreadable.', 'danger')
                return redirect(request.url)
                
            h, w = img_bgr.shape[:2]
            aspect_ratio = f"{w}x{h} ({w/h:.2f}:1)"
            
            # Detect face(s) in the query image
            faces = detect_faces(img_bgr)
            if not faces:
                flash('No face detected in the uploaded image. Please ensure clear face visibility.', 'warning')
                return render_template('recognize.html', no_face=True, aspect_ratio=aspect_ratio)
                
            stored_embeddings = get_all_embeddings()
            faces_visual_data = []
            recognition_results = []
            
            for idx, face in enumerate(faces):
                query_emb = extract_face_embedding(img_bgr, face_info=face)
                match_res = match_face_embedding(query_emb, stored_embeddings, threshold=config.SIMILARITY_THRESHOLD)
                
                face_viz = {
                    'bbox': face['bbox'],
                    'kps': face['kps'],
                    'identity': match_res['name'],
                    'similarity': match_res['similarity'],
                    'status': match_res['status']
                }
                faces_visual_data.append(face_viz)
                recognition_results.append(match_res)
                
            # Draw visual bounding box & landmarks on copy of original image
            annotated_img = draw_visualizations(img_bgr, faces_visual_data)
            
            # Save annotated output file for display
            filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(save_path, annotated_img)
            
            primary_result = recognition_results[0]
            
            return render_template(
                'recognize.html',
                result=primary_result,
                all_results=recognition_results,
                annotated_image_url=url_for('uploaded_file', filename=filename),
                image_dimensions=f"{w} x {h}",
                aspect_ratio=aspect_ratio,
                face_count=len(faces)
            )
            
        except Exception as e:
            app.logger.error("Recognition error: %s", str(e), exc_info=True)
            flash(f"Error processing recognition pipeline: {str(e)}", 'danger')
            return redirect(request.url)
            
    return render_template('recognize.html')

@app.route('/persons')
def persons():
    """Display gallery of all registered persons and sample counts."""
    persons_list = get_all_persons_with_counts()
    return render_template('persons.html', persons=persons_list)

@app.route('/delete/<int:person_id>', methods=['POST'])
def delete_person_route(person_id):
    """Delete a registered person record."""
    delete_person(person_id)
    flash('Registered person record deleted successfully.', 'info')
    return redirect(url_for('persons'))

@app.route('/about')
def about():
    """Technical architecture and documentation page."""
    return render_template('about.html', threshold=config.SIMILARITY_THRESHOLD)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded result images safely."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/stats')
def api_stats():
    """API endpoint for system status metrics."""
    persons = get_all_persons_with_counts()
    return jsonify({
        'status': 'active',
        'total_persons': len(persons),
        'total_samples': sum(p['sample_count'] for p in persons),
        'similarity_threshold': config.SIMILARITY_THRESHOLD
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

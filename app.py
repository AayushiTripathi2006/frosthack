from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
from ultralytics import YOLO
from PIL import Image, ExifTags
import cv2
import pandas as pd
from sort import Sort
from flask_socketio import SocketIO, emit
from collections import defaultdict
from flask_migrate import Migrate
from detection.detect_plastic import PlasticDetector
from tracking.sort_tracker import Sort
from gps_mapping.heatmap import generate_heatmap
from detection.detect_plastic import PlasticDetector


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///test.db"
app.config['SQLALCHEMY_TRACK_MOD'] = False
app.secret_key = 'your_secret_key'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

socketio = SocketIO(app, cors_allowed_origins="*")


# Ensure the directory for images exists
UPLOAD_FOLDER = "./static/uploads/images"
VIDEO_UPLOAD_FOLDER = './uploads'
FRAME_FOLDER = './frames'
OUTPUT_CSV = 'submit.csv'
YOLO_MODEL_PATH = "YOLO_Custom_v8m.pt"
model = YOLO(YOLO_MODEL_PATH)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_UPLOAD_FOLDER'] = VIDEO_UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

# Load YOLOv8 model for plastic detection
plastic_model = YOLO(YOLO_MODEL_PATH)

# Load YOLOv8 model for video processing
video_model_path = 'YOLO_Custom_v8m.pt'
try:
    video_model = YOLO(video_model_path)
    print(f"Loaded YOLO model from: {video_model_path}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")

# Initialize SORT for tracking
tracker = Sort()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    contact = db.Column(db.String(150), nullable=False)
    name = db.Column(db.String(150), nullable=True)
    address = db.Column(db.String(150), nullable=True)
    company = db.Column(db.String(150), nullable=True)
    experience = db.Column(db.String(150), nullable=True)
    profile_pic = db.Column(db.String(150), nullable=True)
    registration_date = db.Column(db.DateTime, default=datetime.utcnow)

class ContactUs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False)
    message = db.Column(db.Text, nullable=False)

class ImageUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(150), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)

class VideoUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Allow null for public uploads
    filename = db.Column(db.String(150), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)

class NGORegistration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False)
    phone = db.Column(db.String(150), nullable=False)
    company = db.Column(db.String(150), nullable=False)
    website = db.Column(db.String(150), nullable=True)
    address = db.Column(db.String(150), nullable=False)
    message = db.Column(db.Text, nullable=False)
    registration_date = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()

@app.route('/ngoregister', methods=['GET', 'POST'])
def ngoregister():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        company = request.form['company']
        website = request.form['website']
        address = request.form['address']
        message = request.form['message']

        new_registration = NGORegistration(
            name=name,
            email=email,
            phone=phone,
            company=company,
            website=website,
            address=address,
            message=message
        )
        db.session.add(new_registration)
        db.session.commit()
        flash("Registration successful!", "success")
        return redirect(url_for('thankyou'))
    return render_template('ngoregister.html')
@app.route('/ngocontact')
def ngocontact():
    registrations = NGORegistration.query.all()
    return render_template('ngocontact.html', registrations=registrations)
@app.route('/thankyou')
def thankyou():
    return render_template('thankyou.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':
            session['user'] = 'admin'
            return redirect(url_for('dashboard'))
        else:
            return 'Invalid credentials'
    return render_template('login.html')
@app.route('/api/registrations')
def api_registrations():
    registrations = User.query.all()
    daily_registrations = defaultdict(int)
    for user in registrations:
        day = user.registration_date.strftime('%Y-%m-%d')
        daily_registrations[day] += 1
    print(daily_registrations)  # Debug print statement
    return jsonify(daily_registrations)


@app.route('/dashboard')
def dashboard():
    if 'user' in session and session['user'] == 'admin':
        users = User.query.all()
        return render_template('dashboard.html', users=users)
    else:
        return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        contact = request.form['contact']
        email = request.form['email']
        confirm_password = request.form['confirm-password']

        if not username or not email or not contact or not password or not confirm_password:
            flash("All fields are required.", "danger")
            return redirect(url_for('register'))
        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('register'))
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists.", "danger")
            return redirect(url_for('register'))
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash("Email already exists.", "danger")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password, contact=contact, email=email)
        db.session.add(new_user)
        db.session.commit()
        socketio.emit('registration_update')
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('user_login'))
    return render_template('register.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        new_contact = ContactUs(name=name, email=email, message=message)
        db.session.add(new_contact)
        db.session.commit()
        flash("Your message has been sent successfully!", "success")
        return redirect(url_for('thank_you'))
    return render_template('contact.html')

@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

@app.route('/user_login', methods=['GET', 'POST'])
def user_login():
    error_message = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user'] = user.username
            flash("Login successful!", "success")
            return redirect('/user_dashboard')
        else:
            error_message = "Invalid credentials. Please check your username and password."
    return render_template('userlogin.html', error_message=error_message)

@app.route('/user_dashboard')
def user_dashboard():
    if 'user' in session:
        current_user = User.query.filter_by(username=session['user']).first()
        image_count = ImageUpload.query.filter_by(user_id=current_user.id).count()
        video_count = VideoUpload.query.filter_by(user_id=current_user.id).count()
        eligible = image_count >= 5 and video_count >= 1
        return render_template('user_dashboard.html', eligible=eligible, image_count=image_count, video_count=video_count)
    else:
        return redirect(url_for('user_login'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user' in session:
        user = User.query.filter_by(username=session['user']).first()
        if request.method == 'POST':
            try:
                user.name = request.form['name']
                user.address = request.form['address']
                user.contact = request.form['contact']
                user.company = request.form['company']
                user.experience = request.form['experience']
                if 'profile_pic' in request.files:
                    profile_pic = request.files['profile_pic']
                    profile_pic_filename = profile_pic.filename
                    profile_pic_path = os.path.join('static/profile_pics', profile_pic_filename)
                    os.makedirs(os.path.dirname(profile_pic_path), exist_ok=True)
                    profile_pic.save(profile_pic_path)
                    user.profile_pic = profile_pic_filename
                db.session.commit()
                flash("Profile updated successfully!", "success")
                return redirect(url_for('profile_details'))
            except Exception as e:
                flash(f"An error occurred: {str(e)}", "danger")
            return redirect(url_for('profile'))
        return render_template('profile.html', user=user)
    else:
        return redirect(url_for('user_login'))

@app.route('/profile_details')
def profile_details():
    if 'user' in session:
        user = User.query.filter_by(username=session['user']).first()
        return render_template('profile_details.html', user=user)
    else:
        return redirect(url_for('user_login'))

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('hello_world'))

@app.route('/registerdashboard')
def register_dashboard():
    if 'user' in session and session['user'] == 'admin':
        users = User.query.all()
        return render_template('registerdashboard.html', users=users)
    else:
        return redirect(url_for('user_login'))



@app.route('/public_upload_video', methods=['GET', 'POST'])
def public_upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file uploaded", "danger")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No selected file", "danger")
            return redirect(request.url)
        video_path = os.path.join(VIDEO_UPLOAD_FOLDER, file.filename)
        file.save(video_path)
        new_video = VideoUpload(
            user_id=None,  # Explicitly set to None for public uploads
            filename=file.filename
        )  # Public upload, no user_id
        db.session.add(new_video)
        db.session.commit()
        return redirect(url_for('process_video', video_name=file.filename))
    return render_template('public_upload_video.html')

@app.route('/process/<video_name>')
def process_video(video_name):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
    frame_index = 0
    df = pd.DataFrame(columns=['Frame', 'Geo_Tag_URL', 'Prediction'])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(FRAME_FOLDER, f"frame_{frame_index:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        results = model.predict(source=frame_filename, save=True, conf=0.37)

        plastic_count = sum(1 for result in results for label in result.names.values() if "plastic" in label.lower())

        prediction_text = "Waste Plastic Detected" if plastic_count > 10 else "No Waste Plastic"
        df.loc[len(df)] = [f"frame_{frame_index:04d}.jpg", "28.6139°N 77.2090°E", prediction_text]

        frame_index += 1

    cap.release()
    df.to_csv(OUTPUT_CSV, index=False)

    return render_template('process_complete.html')
@app.route('/download')
def download_csv():
    return send_file(OUTPUT_CSV, as_attachment=True)

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            upload_folder = os.path.join('static/uploads/images')
            os.makedirs(upload_folder, exist_ok=True)
            file.save(os.path.join(upload_folder, filename))
            user = User.query.filter_by(username=session['user']).first()
            new_image = ImageUpload(user_id=user.id, filename=filename)
            db.session.add(new_image)
            db.session.commit()
            flash('Image uploaded successfully!', 'success')
            return redirect(url_for('user_dashboard'))
    return render_template('upload_image.html')

@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            upload_folder = os.path.join('static/uploads/videos')
            os.makedirs(upload_folder, exist_ok=True)
            file.save(os.path.join(upload_folder, filename))
            user = User.query.filter_by(username=session['user']).first()
            new_video = VideoUpload(user_id=user.id, filename=filename)
            db.session.add(new_video)
            db.session.commit()
            flash('Video uploaded successfully!', 'success')
            return redirect(url_for('user_dashboard'))
    return render_template('upload_video.html')

@app.route('/certificate')
def certificate():
    if 'user' in session:
        user = User.query.filter_by(username=session['user']).first()
        if user:
            date = datetime.now().strftime("%B %d, %Y")
            return render_template('certificate.html', user=user, date=date)
    return redirect(url_for('login'))

@app.route('/contact_us_dashboard')
def contact_us_dashboard():
    if 'user' in session and session['user'] == 'admin':
        contacts = ContactUs.query.all()
        return render_template('contact_us_dashboard.html', contacts=contacts)
    else:
        return redirect(url_for('login'))

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/smartdata')
def smartdata():
    return render_template('smartdata.html')

@app.route('/robocleanup')
def robocleanup():
    return render_template('robocleanup.html')
@app.route('/classify')
def classify():
    return render_template('classify.html')


@app.route('/uploadthe_video', methods=['GET', 'POST'])
def uploadthe_video():
    if request.method == 'POST':
        video_file = request.files['file']
        if video_file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
            video_file.save(video_path)
            return redirect(url_for('process_video', video_name=video_file.filename))
    return render_template('public_upload_video.html')

@app.route('/upload_video2', methods=['GET', 'POST'])
def upload_video2():
    if request.method == 'POST':
        # Check if a file is provided
        file = request.files.get('video')
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process video and get data
            frame_data = process_video2(file_path)

            return render_template('result.html', frame_data=frame_data)
    return render_template('robocleanup.html')

# Mockup prediction function
def process_video2(video_path):
    # Simulate processing and frame data
    frame_data = [("frame_1.jpg", "High Density"), ("frame_2.jpg", "Low Density")]
    return frame_data

def extract_gps_data(image_path):
    try:
        img = Image.open(image_path)
        exif_data = {
            ExifTags.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in ExifTags.TAGS
        }
        if 'GPSInfo' in exif_data:
            gps_info = exif_data['GPSInfo']
            ns = gps_info[2]
            ew = gps_info[4]
            latitude = (((ns[0] * 60) + ns[1]) * 60 + ns[2]) / 60 / 60
            longitude = (((ew[0] * 60) + ew[1]) * 60 + ew[2]) / 60 / 60
            return f"{latitude}°N, {longitude}°E"
        return "No GPS Data Found"
    except Exception as e:
        return f"Error reading GPS data: {e}"
    
@app.route('/public_upload_image', methods=['GET', 'POST'])
def public_upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file uploaded", "danger")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No selected file", "danger")
            return redirect(request.url)
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)
        
        # Run the YOLO prediction (set save=True if you want to save output files)
        results = plastic_model.predict(source=image_path, save=True, conf=0.37)
        gps_info = extract_gps_data(image_path)
        
        # Create a filename for the predicted image
        predicted_image_filename = 'predicted_' + file.filename
        predicted_image_path = os.path.join(UPLOAD_FOLDER, predicted_image_filename)
        results[0].save(predicted_image_path)
        
        # Build a detection result dictionary
        detection_data = {
            "prediction": results[0].boxes if results else "No detections",
            "gps": gps_info,
            "predicted_image": 'uploads/images/' + predicted_image_filename
        }
        
        # Emit the detection results via SocketIO (to stream to any connected client)
        socketio.emit('detection_results', detection_data)
        
        flash("Image uploaded and processed successfully.", "success")
        # Render the same template with the detection data (for non‑real‑time fallback)
        return render_template('public_upload_image.html', **detection_data)
    return render_template('public_upload_image.html')
@app.route('/')
def main():
    video_path = input("Enter the path of the video file: ").strip()
    if not os.path.exists(video_path):
        print(f"Error: {video_path} does not exist!")
        return

    # Initialize Detector and Tracker
    detector = PlasticDetector(YOLO_MODEL_PATH)
    tracker = Sort()

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    all_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_objects(frame)
        tracked_objects = tracker.update(detections)

        # Draw tracking results
        for obj_id, x1, y1, x2, y2 in tracked_objects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save detections for heatmap analysis
        all_detections.extend(tracked_objects)

        # Display frame
        cv2.imshow("Plastic Waste Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Generate heatmap
    generate_heatmap(all_detections)
    print("Processing completed!")



# (Other routes remain unchanged; for brevity, they are not repeated here)
# =============================================================================
# Run the App using SocketIO
# =============================================================================
if __name__ == '__main__':
    socketio.run(app, debug=True)


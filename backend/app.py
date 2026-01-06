import os
import re
import pickle
import numpy as np
import pandas as pd
import subprocess
import threading
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_pymongo import PyMongo
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from bson.objectid import ObjectId
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from logging.handlers import RotatingFileHandler
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='../frontend', static_folder='../Images')
app.secret_key = os.getenv("FLASK_SECRET_KEY")
CORS(app)

if not os.path.exists('logs'): os.mkdir('logs')

file_handler = RotatingFileHandler('logs/edu2job.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)


@app.after_request
def set_security_headers(response):
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response
def sanitize_input(text):
    if isinstance(text, str):
        return re.sub(r'<[^>]*>', '', text).strip()
    return text
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["500 per day", "100 per hour"],
    storage_uri="memory://"
)

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

app.config["MONGO_URI"] = os.getenv("MONGO_URI")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../ml-model')

mongo = PyMongo(app)
jwt = JWTManager(app)
bcrypt = Bcrypt(app)
oauth = OAuth(app)

google = oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"), 
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../ml-model')

model = None
label_encoders = None
scaler = None
mlb = None
feature_names = None
feature_selector = None

def load_artifacts():
    global model, label_encoders, scaler, mlb, feature_names, feature_selector
    try:
        model = pickle.load(open(os.path.join(MODEL_DIR, 'career_model.pkl'), 'rb'))
        label_encoders = pickle.load(open(os.path.join(MODEL_DIR, 'label_encoders.pkl'), 'rb'))
        scaler = pickle.load(open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb'))
        mlb = pickle.load(open(os.path.join(MODEL_DIR, 'skills_mlb.pkl'), 'rb'))
        feature_names = pickle.load(open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'rb'))
        
        try:
            feature_selector = pickle.load(open(os.path.join(MODEL_DIR, 'feature_selector.pkl'), 'rb'))
        except:
            feature_selector = None

        print("Model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading ML: {e}")
        return False
load_artifacts()

#Background Training
def run_training_script():
    try:
        script_path = os.path.join(MODEL_DIR, 'train_model.py')
        subprocess.run(["python", script_path], check=True)
        print("Training complete. Reloading model...")
        load_artifacts()
    except Exception as e:
        print(f"Background Training Failed: {e}")

def get_processed_vector(data):
    global label_encoders, scaler, mlb, feature_names

    skills_list = data.get("skills", [])
    if not isinstance(skills_list, list): skills_list = []
    STOP_WORDS = {"communication", "problem solving", "critical thinking", "teamwork", "leadership"}
    clean_skills = [s.strip() for s in skills_list if s.strip().lower() not in STOP_WORDS]
    valid_skills = [skill for skill in clean_skills if skill in mlb.classes_]

    #Safe Label Encoding
    def safe_transform(encoder, value):
        try:
            return int(encoder.transform([value])[0]) 
        except ValueError:
            return int(encoder.transform([encoder.classes_[0]])[0])

    degree_enc = safe_transform(label_encoders["degree"], data.get("degree", "B.Tech"))
    spec_enc = safe_transform(label_encoders["specialization"], data.get("specialization", "Computer Science"))
    cert_enc = safe_transform(label_encoders["certifications"], data.get("certifications", "None"))

    # Normalization (Scaling)
    try:
        val_cgpa = float(data.get("cgpa", 0))
        val_year = int(data.get("graduation_year", 2024))
    except:
        val_cgpa, val_year = 0.0, 2024
    
    # Scale using the loaded Scaler
    numeric_df = pd.DataFrame([[val_cgpa, val_year]], columns=["cgpa", "graduation_year"])
    scaled_vals = scaler.transform(numeric_df)
    
    #Internship Handling
    raw_intern = data.get("internships", 0)
    intern_val = 1 if str(raw_intern).lower() in ['1', 'true', 'yes'] else 0

    #Construct Final Vector Dictionary
    processed_record = {
        "cgpa_scaled": float(scaled_vals[0][0]),
        "graduation_year_scaled": float(scaled_vals[0][1]),
        "internship_encoded": intern_val,
        "degree_encoded": degree_enc,
        "specialization_encoded": spec_enc,
        "certifications_encoded": cert_enc,
        "skills_encoded_count": len(valid_skills) 
    }
    
    return processed_record, valid_skills

#Routes
@app.route('/')
def home(): return render_template('index.html')

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')

@app.route('/admin')  
def admin_dashboard(): return render_template('admin_dashboard.html')

#Config API
@app.route('/api/config', methods=['GET'])
def get_config():
    try:
        path = os.path.join(MODEL_DIR, 'metadata.pkl')
        if os.path.exists(path):
            metadata = pickle.load(open(path, 'rb'))
            return jsonify(metadata), 200
        return jsonify({"message": "Metadata not found. Train model first."}), 404
    except Exception as e:
        return jsonify({"message": str(e)}), 500

#Auth
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = sanitize_input(data.get('name', ''))
    email = sanitize_input(data.get('email', ''))
    password = data.get('password', '')

    if len(password) < 8:
        return jsonify({"message": "Password must be at least 8 characters long"}), 400
    if not re.search(r"\d", password):
        return jsonify({"message": "Password must contain at least one number"}), 400
    if not re.search(r"[A-Z]", password):
        return jsonify({"message": "Password must contain at least one uppercase letter"}), 400

    if not re.fullmatch(r"^[^@]+@[^@]+\.[^@]+$", email):
        return jsonify({"message": "Invalid email format"}), 400

    if mongo.db.users.find_one({"email": email}):
        return jsonify({"message": "Email already exists"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    try:
        mongo.db.users.insert_one({
            "name": name,
            "email": email,
            "password": hashed_password,
            "role": "student",
            "degree": "", "specialization": "", "cgpa": "", 
            "graduation_year": "", "certifications": "", "skills": [], "internships": 0
        })
        app.logger.info(f"New user registered: {email}")
        return jsonify({"message": "Registration successful"}), 201
    except Exception as e:
        app.logger.error(f"Registration Error: {e}") 
        return jsonify({"message": "Server Error"}), 500


@app.route('/login', methods=['POST'])
@limiter.limit("25 per minute")
def login():
    data = request.get_json()
    user = mongo.db.users.find_one({"email": data['email']})

    if user:
        if not user.get('password'):
            return jsonify({"message": "Use Google Login."}), 400
        
        if bcrypt.check_password_hash(user['password'], data['password']):
            token = create_access_token(identity=str(user['_id']))
            role = user.get('role', 'student')
            return jsonify({"token": token, "name": user['name'], "role": role}), 200
    return jsonify({"message": "Invalid credentials"}), 401

@app.route('/login/google')
def google_login():
    redirect_uri = url_for('google_authorize', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/login/google/callback')
def google_authorize():
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo') or google.userinfo()
        email, name = user_info['email'], user_info['name']
        
        user = mongo.db.users.find_one({"email": email})
        if not user:
            user_id = mongo.db.users.insert_one({
                "name": name, "email": email, "password": "", "role": "student",
                "degree": "", "specialization": "", "cgpa": "", "graduation_year": "", 
                "certifications": "", "skills": [], "internships": 0
            }).inserted_id
        else:
            user_id = user['_id']
        jwt_token = create_access_token(identity=str(user_id))
        return render_template('google_callback.html', token=jwt_token)
    except Exception as e:
        print(f"Google Login Error: {e}")
        return redirect('/')

#Profile & Prediction
@app.route('/api/profile', methods=['GET'])
@jwt_required()
def get_profile():
    user = mongo.db.users.find_one({"_id": ObjectId(get_jwt_identity())})
    user['_id'] = str(user['_id'])
    return jsonify(user), 200

@app.route('/api/update_profile', methods=['PUT', 'PATCH'])
@jwt_required()
def update_profile():
    user_id = get_jwt_identity()
    data = request.get_json()
    if 'cgpa' in data:
        try:
            val = float(data['cgpa'])
            if val < 0 or val > 10: return jsonify({"message": "CGPA must be 0-10"}), 400
        except: return jsonify({"message": "Invalid CGPA"}), 400

    if 'graduation_year' in data:
        try:
            year = int(data['graduation_year'])
            if year < 2020 or year > 2030: 
                return jsonify({"message": "Invalid Graduation Year"}), 400
        except ValueError:
            return jsonify({"message": "Year must be a number"}), 400
        
    if 'degree' in data and not data['degree'].strip():
        return jsonify({"message": "Degree cannot be empty"}), 400
    try:
        processed_vector, _ = get_processed_vector(data)
    except Exception as e:
        print(f"Preprocessing Error: {e}")
        return jsonify({"message": "Error processing data for model"}), 500
    update_doc = {
        "$set": { 
            **data,
            "education_processed": processed_vector
        }
    }
    mongo.db.users.update_one({"_id": ObjectId(user_id)},update_doc)
    return jsonify({"message": "Profile updated Successfully!& data preprocessed successfully!"}), 200
    

@app.route('/api/predict', methods=['POST'])
@jwt_required()
def predict():
    global model, label_encoders, scaler, mlb, feature_names, feature_selector
    if not model:
        if not load_artifacts():
            return jsonify({"message": "Model not loaded. Train first."}), 503
    user_id = get_jwt_identity()
    data = request.get_json()

    try:
        processed_record, valid_skills = get_processed_vector(data)
        skills_matrix = mlb.transform([valid_skills])
        skills_encoded_df = pd.DataFrame(skills_matrix, columns=mlb.classes_)
        skills_dict = skills_encoded_df.iloc[0].to_dict()
        input_record = {
            "cgpa": processed_record['cgpa_scaled'],
            "graduation_year": processed_record['graduation_year_scaled'],
            "internship_experience": processed_record['internship_encoded'],
            "degree": processed_record['degree_encoded'],
            "specialization": processed_record['specialization_encoded'],
            "certifications": processed_record['certifications_encoded'],
        }
        input_record.update(skills_dict)
        input_df = pd.DataFrame([input_record])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        #Predict
        probs = model.predict_proba(input_df)[0]
        top3_idx = np.argsort(probs)[-3:][::-1]
        results = []
        for idx in top3_idx:
            job_category = label_encoders["job_role"].inverse_transform([idx])[0]
            clean_job_category = job_category.replace('/', ' ')
            confidence = round(float(probs[idx]) * 100, 1)
            results.append({"job_role": clean_job_category, "confidence": confidence})

        if not results:
            results.append({"job_role": "Uncertain", "confidence": 0})
        
        #Save to History
        user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
        mongo.db.history.insert_one({
            "user_id": ObjectId(user_id),
            "user_name": user.get("name", "Unknown"),
            "prediction": results[0]["job_role"],
            "confidence": results[0]["confidence"],
            "top_predictions": results,
            "date": datetime.now()
        })

        return jsonify({"top_predictions": results, "justification": f"Based on your skills: {', '.join(valid_skills)}" }), 200

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"message": f"Server Error: {str(e)}"}), 500

#History & Feedback
@app.route('/api/history', methods=['GET'])
@jwt_required()
def history():
    user_id = get_jwt_identity()
    cursor = mongo.db.history.find({"user_id": ObjectId(user_id)}).sort("date", -1)
    history_list = []
    for h in cursor:
        history_list.append({
            "_id": str(h["_id"]),
            "prediction": h.get("prediction", "Unknown"),
            "confidence": h.get("confidence", 0),
            "top_predictions": h.get("top_predictions", []),
            "feedback": h.get("feedback", None),
            "date": h["date"].strftime("%Y-%m-%d %H:%M")
        })
    return jsonify(history_list), 200

@app.route('/api/feedback', methods=['POST'])
@jwt_required()
def submit_feedback():
    data = request.get_json()
    pred_id = data.get('prediction_id')
    rating = int(data.get('rating', 0))
    user_id = get_jwt_identity()

    result = mongo.db.history.update_one(
        {"_id": ObjectId(pred_id), "user_id": ObjectId(user_id)},
        {"$set": {"feedback": rating}}
    )
    if result.modified_count == 1:
        return jsonify({"message": "Feedback saved!"}), 200
    return jsonify({"message": "Error saving feedback"}), 404

@app.route('/api/reset_password_manual', methods=['POST'])
def reset_password_manual():
    data = request.get_json()
    user = mongo.db.users.find_one({"email": data.get('email'), "name": data.get('name')})
    if not user:
        return jsonify({"message": "User not found."}), 404
    hashed_new = bcrypt.generate_password_hash(data.get('new_password')).decode('utf-8')
    mongo.db.users.update_one({"_id": user['_id']}, {"$set": {"password": hashed_new}})
    return jsonify({"message": "Password reset successful!"}), 200

@app.route('/api/change_password', methods=['POST'])
@jwt_required()
def change_password():
    data = request.get_json()
    user_id = get_jwt_identity()
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})

    if not bcrypt.check_password_hash(user['password'], data.get('old_password')):
        return jsonify({"message": "Invalid old password"}), 401
    hashed_new = bcrypt.generate_password_hash(data.get('new_password')).decode('utf-8')
    mongo.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"password": hashed_new}})
    return jsonify({"message": "Password changed successfully!"}), 200

#Stats & Admin
@app.route('/api/stats/job_distribution', methods=['GET'])
@jwt_required()
def job_distribution():
    pipeline = [
        {"$group": {"_id": "$prediction", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    stats = list(mongo.db.history.aggregate(pipeline))
    labels = [item['_id'] for item in stats]
    data = [item['count'] for item in stats]
    return jsonify({"labels": labels, "data": data}), 200

@app.route('/api/stats/comparison', methods=['GET'])
@jwt_required()
def compare_stats():
    user_id = get_jwt_identity()
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    try:
        user_cgpa = float(user.get('cgpa', 0) if user.get('cgpa') else 0)
        user_skills_count = len(user.get('skills', []) if isinstance(user.get('skills'), list) else [])
    except:
        user_cgpa, user_skills_count = 0, 0

    user_degree = user.get('degree', '')  
    pipeline = [
        {"$match": {"degree": user_degree}}, 
        {
            "$group": {
                "_id": None,
                "avg_cgpa": { "$avg": { "$convert": { "input": "$cgpa", "to": "double", "onError": 0, "onNull": 0 } } },
                "avg_skills": { "$avg": { "$size": { "$ifNull": ["$skills", []] } } }
            }
        }
    ]

    agg = list(mongo.db.users.aggregate(pipeline))
    market_stats = [7.5, 6] 
    if agg:
        market_stats = [
            round(agg[0].get('avg_cgpa', 7.5) or 7.5, 2),
            round(agg[0].get('avg_skills', 6) or 6, 1)
        ]
    insights = []
    if user_cgpa > market_stats[0]:
        insights.append(f"🎉 Your CGPA ({user_cgpa}) is higher than the average {user_degree} student ({market_stats[0]}).")
    else:
        insights.append(f"📈 The average CGPA for {user_degree} students is {market_stats[0]}. Keep pushing!")

    return jsonify({
        "labels": ["CGPA", "Skills"],
        "user_data": [user_cgpa, user_skills_count],
        "market_data": market_stats,
        "insights": insights  
    }), 200

@app.route('/api/stats/degree_job', methods=['GET'])
@jwt_required()
def degree_job_stats():
    pipeline = [
        {"$lookup": {"from": "users", "localField": "user_id", "foreignField": "_id", "as": "user_info"}},
        {"$unwind": "$user_info"},
        {"$group": {"_id": "$user_info.degree", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    data = list(mongo.db.history.aggregate(pipeline))
    labels = [d['_id'] if d['_id'] else "Unknown" for d in data]
    values = [d['count'] for d in data]
    return jsonify({"labels": labels, "data": values}), 200

@app.route('/api/admin/users', methods=['GET'])
@jwt_required()
def get_all_users():
    user_id = get_jwt_identity()
    admin = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    if admin.get('role') != 'admin': 
        return jsonify({"message": "Access Denied"}), 403

    users_cursor = mongo.db.users.find({"role": {"$ne": "admin"}})
    user_list = []
    for u in users_cursor:
        join_date = u['_id'].generation_time.strftime("%Y-%m-%d") if u.get('_id') else "Unknown"
        user_list.append({"name": u.get('name','Unknown'), "email": u.get('email','Unknown'), "date": join_date})
    return jsonify(user_list), 200

@app.route('/api/admin/stats', methods=['GET'])
@jwt_required()
def admin_stats():
    user_id = get_jwt_identity()
    admin = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    if admin.get('role') != 'admin': 
        return jsonify({"message": "Access Denied"}), 403

    total_users = mongo.db.users.count_documents({"role": "student"})
    total_preds = mongo.db.history.count_documents({})

    feedback_agg = list(mongo.db.history.aggregate([
        {"$match": {"feedback": {"$exists": True}}},
        {"$group": {"_id": None, "avg_rating": {"$avg": "$feedback"}}}
    ]))
    avg_rating = round(feedback_agg[0]['avg_rating'], 1) if feedback_agg else "N/A"

    recent_logs = list(mongo.db.history.find().sort("date", -1).limit(10))
    logs_data = []
    for log in recent_logs:
        logs_data.append({
            "_id": str(log["_id"]),
            "user_name": log.get("user_name", "Unknown"),
            "prediction": log.get("prediction", "N/A"),
            "confidence": log.get("confidence", 0),
            "feedback": log.get("feedback", "-"),
            "flagged": log.get("flagged", False),
            "date": log["date"].strftime("%Y-%m-%d %H:%M")
        })

    return jsonify({
        "total_users": total_users,
        "total_predictions": total_preds,
        "average_rating": avg_rating,
        "recent_logs": logs_data
    }), 200

@app.route('/api/admin/flag_prediction', methods=['POST'])
@jwt_required()
def flag_prediction():
    user = mongo.db.users.find_one({"_id": ObjectId(get_jwt_identity())})
    if user.get('role') != 'admin': 
        return jsonify({"message": "Access Denied"}), 403

    data = request.get_json()
    pred_id = data.get('prediction_id')
    
    result = mongo.db.history.update_one(
        {"_id": ObjectId(pred_id)},
        {"$set": {"flagged": True}}
    )
    
    if result.modified_count == 1:
        return jsonify({"message": "Prediction flagged for review."}), 200
    return jsonify({"message": "Error flagging prediction."}), 400

@app.route('/api/admin/upload_dataset', methods=['POST'])
@jwt_required()
def upload_dataset():
    user = mongo.db.users.find_one({"_id": ObjectId(get_jwt_identity())})
    if user.get('role') != 'admin': return jsonify({"message": "Access Denied"}), 403

    if 'file' not in request.files: return jsonify({"message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"message": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            required_cols = {'degree', 'specialization', 'cgpa', 'graduation_year', 'skills', 'job_role'}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                return jsonify({"message": f"Invalid CSV. Missing columns: {missing}"}), 400
            
            filename = "career_dataset.csv"
            file.seek(0)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      
            thread = threading.Thread(target=run_training_script)
            thread.start()
            return jsonify({"message": "Dataset uploaded & Training started."}), 200
        except Exception as e:
            return jsonify({"message": f"Error processing file: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
Edu2Job - AI-Powered Career Prediction System 

📌 Project Overview

Edu2Job is a full-stack web application that uses machine learning to predict suitable job roles for students based on their academic profiles, skills, and experience. The system provides personalized career recommendations with confidence scores and comparative market insights.

✨ Key Features

🔐 Authentication & Security

JWT-based authentication with secure token storage

Google OAuth 2.0 integration for social login

Password encryption using Flask-Bcrypt

Rate limiting (100 requests/hour per IP)

Security headers (XSS protection, CSRF prevention)

📊 ML-Powered Prediction Engine

XGBoost classifier trained on career dataset

Multi-label skills encoding with noise reduction

Intelligent preprocessing of academic data

Top-3 predictions with confidence percentages

Justification system explaining why roles were recommended

👤 User Dashboard

Profile management with academic details

Interactive prediction history with feedback system

Market comparison charts (CGPA vs market average)

Trending job visualization using Chart.js

Real-time skill tagging with autocomplete

👑 Admin Panel

User management with registration tracking

System statistics dashboard

Prediction flagging for incorrect results

CSV dataset upload for model retraining

Training logs and performance metrics

📈 Analytics & Insights

Job distribution charts by degree/role

Personal vs market benchmark comparisons

Feedback aggregation (1-5 star ratings)

Degree-wise prediction statistics

🏗️ Tech Stack

Backend

Flask - Python web framework

PyMongo - MongoDB database integration

Flask-JWT-Extended - Secure authentication

Scikit-learn & XGBoost - Machine learning

Pandas & NumPy - Data processing

Authlib - OAuth integration

Frontend

HTML5, CSS3, JavaScript (Vanilla)

Chart.js - Data visualization

Font Awesome - Icons

Google Fonts (Poppins) - Typography

Database

MongoDB - NoSQL database for user data and prediction history

Deployment & Infrastructure

Environment variables via python-dotenv

Logging with RotatingFileHandler

CORS enabled for API security

Model artifact backup system

📁 Project Structure
Edu2Job/ 
├── backend/ 
│   ├── logs/                     # Application logs
│   ├── app.py                    #Main Flask Application   
│   ├── seed_data.py             # Database Seeder Script
│   └── .env 
├── frontend/              # HTML templates
│   ├── index.html         # Login/Registration
│   ├── dashboard.html      # User dashboard
│   ├── google_callback.html     # OAuth callback page     
│   └── admin_dashboard.html     #User Dashboard                     
├── ml-model/               # ML artifacts directory
│   ├── backups/
│   ├── train_model.py      # ML model training script
│   ├── career_model.pkl
│   ├── label_encoders.pkl
│   ├── scaler.pkl
│   ├── skills_mlb.pkl
│   ├── feature_names.pkl
│   ├── feature_selectors.pkl 
│   ├── metadata.pkl
│   └── career_dataset.csv  # Training data  
├── Images/
├── .gitignore
├── requirements.txt 
└── README.md              # Model backups

    
🚀 Installation & Setup

Prerequisites

Python 3.8+

MongoDB instance (local or Atlas)

Google OAuth credentials

1. Clone Repository

bash

git clone <repository-url>

cd Edu2Job

2. Install Dependencies

bash

pip install -r requirements.txt

3. Configure Environment Variables
   
Create a .env file in the root directory:

env

FLASK_SECRET_KEY=your_secret_key_here

MONGO_URI=mongodb://localhost:27017/edu2job

JWT_SECRET_KEY=your_jwt_secret

GOOGLE_CLIENT_ID=your_google_client_id

GOOGLE_CLIENT_SECRET=your_google_client_secret

4. Prepare ML Model

bash

# Ensure career_dataset.csv is in ml-model directory

cd ml-model

python train_model.py

5. Run Application
   
bash

python app.py

The application will be available at http://localhost:5000 

📊 Model Training

Dataset Requirements

The CSV file must contain these columns:

degree, specialization, cgpa, graduation_year

skills (comma-separated), job_role (target)

certifications, internship_experience

Training Process

Upload CSV via admin panel

Automatic preprocessing (encoding, scaling)

XGBoost training with evaluation metrics

Artifacts saved with timestamped backups

Model reloaded without restarting app

🔧 API Endpoints

Authentication

POST /register - User registration

POST /login - User login (JWT)

GET /login/google - Google OAuth

POST /api/change_password - Password change

User Operations

GET /api/profile - Get user profile

PUT /api/update_profile - Update profile

POST /api/predict - Get job predictions

GET /api/history - Prediction history

POST /api/feedback - Submit feedback

Analytics

GET /api/stats/job_distribution - Job trends

GET /api/stats/comparison - Market comparison

GET /api/stats/degree_job - Degree statistics

Admin Operations

GET /api/admin/users - List all users

GET /api/admin/stats - System statistics

POST /api/admin/flag_prediction - Flag incorrect predictions

POST /api/admin/upload_dataset - Upload new training data

🎯 Usage Guide

For Students

Register/Login using email or Google

Complete profile with academic details

Add skills focusing on technical competencies

Get predictions with confidence scores

View history and provide feedback

Compare with market benchmarks

For Administrators

Access /admin with admin credentials

Monitor system statistics

Manage users and predictions

Flag incorrect predictions for review

Upload new datasets for model improvement

🔒 Security Features

Input sanitization against XSS attacks

Password validation (min 8 chars, uppercase, number)

JWT token expiration management

Rate limiting on login endpoints

Secure headers (X-Frame-Options, XSS-Protection)

MongoDB injection prevention via PyMongo

📈 Performance Optimization

Background model training using threading

Chart.js for client-side rendering

Efficient ML preprocessing pipelines

MongoDB indexing for faster queries

Artifact caching for model predictions

🐛 Troubleshooting

Common Issues

Model not loading: Check if ml-model/ contains required .pkl files

MongoDB connection failed: Verify MONGO_URI in .env

Google login not working: Ensure OAuth credentials are correct

Prediction errors: Check if skills exist in trained vocabulary

Logs

Application logs: logs/edu2job.log

Training logs: Console output during model training

Backup logs: Check backups/ directory for previous models

📝 Future Enhancements

Real-time notifications for new job matches

Resume parser for automatic profile creation

Company matching based on skills

Interview preparation module

Mobile application (React Native/Flutter)

Advanced analytics with predictive trends

Multi-language support

🤝 Contributing

Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add AmazingFeature')

Push to branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

XGBoost team for the powerful ML library

Flask community for excellent documentation

Chart.js for beautiful data visualization

Font Awesome for icons

Google Fonts for typography

📞 Support
For support, email: [nandinisurapureddy4@gmail.com] or create an issue in the GitHub repository.

Note: This system is for educational purposes. Always verify career advice with professional counselors.

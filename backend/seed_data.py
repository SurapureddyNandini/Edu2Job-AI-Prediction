import pymongo
from flask import Flask
from flask_bcrypt import Bcrypt
from datetime import datetime, timedelta
import random

#CONFIGURATION
MONGO_URI = "mongodb://localhost:27017/jobrole_db"
ADMIN_EMAIL = "example@gmail.com"   #Add your Admin email id
ADMIN_PASS = "password"         #Add your password 

#SETUP FLASK CONTEXT FOR BCRYPT
app = Flask(__name__)
bcrypt = Bcrypt(app)

#DATABASE CONNECTION
client = pymongo.MongoClient(MONGO_URI)
db = client.jobrole_db

#SAMPLE DATASETS
DEGREES = ["B.Tech", "M.Tech", "B.Sc", "MBA", "BCA", "MCA"]
SPECIALIZATIONS = {
    "B.Tech": ["Computer Science", "Information Technology", "Mechanical", "Civil", "Electronics"],
    "M.Tech": ["Data Science", "Artificial Intelligence", "VLSI"],
    "B.Sc": ["Mathematics", "Physics", "Statistics"],
    "MBA": ["Marketing", "Finance", "HR"],
    "BCA": ["Computer Applications"],
    "MCA": ["Computer Applications"]
}
SKILLS_POOL = ["Python", "Java", "SQL", "React", "Machine Learning", "Data Analysis", "AWS", "Excel", "Public Speaking", "Project Management"]
JOB_ROLES = ["Data Scientist", "Software Engineer", "Product Manager", "Business Analyst", "Web Developer", "Mechanical Engineer"]

def create_admin():
    # Check if admin already exists to avoid duplicates
    if db.users.find_one({"email": ADMIN_EMAIL}):
        print("Admin user already exists. Skipping creation.")
        return

    print("Creating Admin user...")
    pw_hash = bcrypt.generate_password_hash(ADMIN_PASS).decode('utf-8')
    db.users.insert_one({
        "name": "Super Admin",
        "email": ADMIN_EMAIL,
        "password": pw_hash,
        "role": "admin",
        "date": datetime.now()
    })
    print(f"   -> Admin created: {ADMIN_EMAIL} / {ADMIN_PASS}")

def create_students(count=20):
    print(f"Appending {count} dummy students...")
    users = []
    for i in range(count):
        deg = random.choice(DEGREES)
        spec = random.choice(SPECIALIZATIONS.get(deg, ["General"]))
        unique_id = random.randint(1000, 9999)
        
        student = {
            "name": f"Student {unique_id}",
            "email": f"student{unique_id}@test.com",
            "password": bcrypt.generate_password_hash("pass123").decode('utf-8'),
            "role": "student",
            "degree": deg,
            "specialization": spec,
            "cgpa": round(random.uniform(6.0, 9.8), 2),
            "graduation_year": random.randint(2023, 2026),
            "certifications": random.choice(["AWS Certified", "Google Data Analytics", "None", "PMP"]),
            "skills": random.sample(SKILLS_POOL, k=random.randint(2, 5)),
            "internships": random.choice([0, 1]),
            "date": datetime.now() - timedelta(days=random.randint(1, 365))
        }
        users.append(student)
    
    res = db.users.insert_many(users)
    print(f"Added {len(res.inserted_ids)} new students.")
    return list(res.inserted_ids)

def create_history(user_ids):
    print("Generating prediction history logs for new users...")
    logs = []

    for uid in user_ids:
        for _ in range(random.randint(1, 3)):
            role = random.choice(JOB_ROLES)
            confidence = round(random.uniform(75.0, 98.0), 1)
            
            # Simulate some feedback and flagging
            feedback = random.choice([None, 3, 4, 5])
            flagged = random.choice([True, False, False, False, False])
            
            log = {
                "user_id": uid,
                "user_name": db.users.find_one({"_id": uid})["name"],
                "prediction": role,
                "confidence": confidence,
                "top_predictions": [
                    {"job_role": role, "confidence": confidence},
                    {"job_role": random.choice(JOB_ROLES), "confidence": round(confidence - 10, 1)}
                ],
                "feedback": feedback,
                "flagged": flagged,
                "date": datetime.now() - timedelta(days=random.randint(0, 60))
            }
            logs.append(log)
            
    db.history.insert_many(logs)
    print(f"Added {len(logs)} history records.")

if __name__ == "__main__":
    create_admin()
    uids = create_students(25) 

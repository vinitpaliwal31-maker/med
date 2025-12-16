import streamlit as st
import sqlite3
import os
import pickle
import json
import re
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="Med4Me - Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
    }
    .stButton>button {
        background: linear-gradient(135deg, #3282b8, #0f4c75);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 700;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(50, 130, 184, 0.4);
    }
    .rec-section {
        background: rgba(15, 76, 117, 0.1);
        border-left: 3px solid #3282b8;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    .rec-section.warning {
        border-left-color: #e74c3c;
        background: rgba(231, 76, 60, 0.1);
    }
    div[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05);
    }
    /* Center login form */
    .block-container {
        max-width: 500px;
        padding-top: 3rem;
    }
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
    }
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
    }
    .stTextInput>div>div>input:focus {
        border-color: #3282b8;
        box-shadow: 0 0 0 1px #3282b8;
    }
    /* Center title */
    h1, h2, h3 {
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px 30px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3282b8, #0f4c75);
    }
</style>
""", unsafe_allow_html=True)

# Database setup
basedir = Path(__file__).parent
DB_PATH = basedir / 'med4me.db'

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = None
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'step' not in st.session_state:
    st.session_state.step = 0

# Database functions
def init_db():
    """Initialize database tables"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Users table
    cur.execute('''CREATE TABLE IF NOT EXISTS user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Doctor-Patient mapping
    cur.execute('''CREATE TABLE IF NOT EXISTS doctor_patient (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_id INTEGER NOT NULL,
        patient_id TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(doctor_id, patient_id),
        FOREIGN KEY (doctor_id) REFERENCES user(id)
    )''')
    
    # Visits table
    cur.execute('''CREATE TABLE IF NOT EXISTS visit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        doctor_id INTEGER,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        symptoms TEXT,
        age TEXT,
        gender TEXT,
        genetic_history TEXT,
        medicine TEXT,
        diagnosis TEXT,
        lifestyle TEXT,
        follow_up TEXT,
        ml_prediction TEXT,
        ml_confidence REAL,
        FOREIGN KEY (doctor_id) REFERENCES user(id)
    )''')
    
    # Create default admin user
    try:
        cur.execute("SELECT * FROM user WHERE username = 'admin'")
        if not cur.fetchone():
            password_hash = generate_password_hash('admin123')
            cur.execute("INSERT INTO user (username, password_hash) VALUES (?, ?)", 
                       ('admin', password_hash))
    except:
        pass
    
    conn.commit()
    conn.close()

# Load ML Model
@st.cache_resource
def load_ml_model():
    """Load ML model if available"""
    try:
        model_path = basedir / 'ml_model.pkl'
        vectorizer_path = basedir / 'vectorizer.pkl'
        treatment_path = basedir / 'treatment_db.json'
        
        if model_path.exists() and vectorizer_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            treatment_db = {}
            if treatment_path.exists():
                with open(treatment_path, 'r') as f:
                    treatment_db = json.load(f)
            return model, vectorizer, treatment_db, True
    except Exception as e:
        st.sidebar.warning(f"ML Model not found: {e}")
    return None, None, {}, False

ML_MODEL, VECTORIZER, TREATMENT_DB, USE_ML = load_ml_model()

# Authentication functions
def authenticate_user(username, password):
    """Authenticate user credentials"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash FROM user WHERE username = ?", (username,))
    result = cur.fetchone()
    conn.close()
    
    if result and check_password_hash(result[1], password):
        return result[0]
    return None

def register_user(username, password):
    """Register new user"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        password_hash = generate_password_hash(password)
        cur.execute("INSERT INTO user (username, password_hash) VALUES (?, ?)", 
                   (username, password_hash))
        conn.commit()
        user_id = cur.lastrowid
        conn.close()
        return user_id, None
    except sqlite3.IntegrityError:
        conn.close()
        return None, "Username already exists"
    except Exception as e:
        conn.close()
        return None, str(e)

# Patient management functions
def get_doctor_patients(doctor_id):
    """Get all patients for a doctor"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT dp.patient_id, dp.created_at,
               (SELECT COUNT(*) FROM visit WHERE doctor_id = ? AND patient_id = dp.patient_id) as visit_count,
               (SELECT date FROM visit WHERE doctor_id = ? AND patient_id = dp.patient_id ORDER BY date DESC LIMIT 1) as last_visit,
               (SELECT symptoms FROM visit WHERE doctor_id = ? AND patient_id = dp.patient_id ORDER BY date DESC LIMIT 1) as last_symptoms
        FROM doctor_patient dp
        WHERE dp.doctor_id = ?
        ORDER BY dp.created_at DESC
    """, (doctor_id, doctor_id, doctor_id, doctor_id))
    patients = cur.fetchall()
    conn.close()
    return patients

def get_patient_history(patient_id, doctor_id):
    """Get patient visit history"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT v.*, u.username 
        FROM visit v
        LEFT JOIN user u ON v.doctor_id = u.id
        WHERE v.patient_id = ? AND v.doctor_id = ?
        ORDER BY v.date ASC
    """, (patient_id, doctor_id))
    history = cur.fetchall()
    conn.close()
    return history

def add_doctor_patient_mapping(doctor_id, patient_id):
    """Add doctor-patient mapping"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("INSERT OR IGNORE INTO doctor_patient (doctor_id, patient_id) VALUES (?, ?)",
                   (doctor_id, patient_id))
        conn.commit()
    except:
        pass
    conn.close()

# ML Recommendation function
def ml_recommendation(symptoms, age, gender, genetic_history=None):
    """Generate medical recommendation using ML or fallback"""
    if USE_ML and ML_MODEL and VECTORIZER:
        try:
            text_vector = VECTORIZER.transform([symptoms.lower()])
            age_val = int(age) if str(age).isdigit() else 30
            gender_val = 0 if gender.lower() in ['male', 'm'] else 1
            
            X = np.hstack([
                text_vector.toarray(),
                np.array([[age_val]]),
                np.array([[gender_val]])
            ])
            
            prediction = ML_MODEL.predict(X)[0]
            probabilities = ML_MODEL.predict_proba(X)[0]
            confidence = float(max(probabilities))
            
            treatment = TREATMENT_DB.get(prediction, TREATMENT_DB.get('general', {}))
            
            diagnosis_map = {
                'fever': 'Acute Febrile Illness',
                'diabetes': 'Type 2 Diabetes Mellitus',
                'cold': 'Upper Respiratory Tract Infection (URTI)',
                'headache': 'Tension Headache / Migraine',
                'hypertension': 'Hypertension',
                'asthma': 'Asthma',
                'gastric': 'Gastritis',
                'allergy': 'Allergic Reaction',
                'arthritis': 'Osteoarthritis',
                'mental_health': 'Anxiety/Depression - Requires Specialist',
                'general': 'General Symptomatic Care'
            }
            
            diagnosis = diagnosis_map.get(prediction, 'Condition Requiring Further Assessment')
            
            return {
                "Diagnosis": diagnosis,
                "Medicine": treatment.get("Medicine", "Symptomatic treatment recommended"),
                "Alternative": treatment.get("Alternative", "Consult specialist for alternatives"),
                "Lifestyle": treatment.get("Lifestyle", "Healthy lifestyle, adequate rest"),
                "Red Flags": treatment.get("Red Flags", "Worsening symptoms, no improvement in 3 days"),
                "Follow-Up": treatment.get("Follow-Up", "Review in 48-72 hours"),
                "Notes": f"ML Model Prediction: {prediction} (Confidence: {confidence*100:.1f}%). This is an AI-assisted recommendation.",
                "ml_prediction": prediction,
                "ml_confidence": confidence
            }
        except Exception as e:
            st.error(f"ML prediction error: {e}")
    
    return fallback_recommendation(symptoms, age, gender, genetic_history)

def fallback_recommendation(symptoms, age, gender, genetic_history=None):
    """Rule-based fallback recommendation"""
    s = (symptoms or "").lower()
    age_val = int(age) if str(age).isdigit() else 30
    
    rec = {
        "Diagnosis": "General symptomatic care",
        "Medicine": "- Paracetamol 500 mg: Every 6 hours as needed",
        "Alternative": "- Ibuprofen 200 mg: Every 6-8 hours if no contraindications",
        "Lifestyle": "Hydration, rest, monitor symptoms.",
        "Red Flags": "Persistent symptoms more than 3 days, high fever, severe pain.",
        "Follow-Up": "Review in 48-72 hours if no improvement.",
        "Notes": "This is a rule-based recommendation. Final prescription authority lies with the licensed physician.",
        "ml_prediction": None,
        "ml_confidence": None
    }
    
    # Fever and infections
    if re.search(r"\bfever\b|\btemperature\b|\bpyrexia\b", s):
        rec.update({
            "Diagnosis": "Acute febrile illness",
            "Medicine": "- Paracetamol 500 mg: Every 6 hours for fever\n- Maintain hydration with ORS",
            "Alternative": "- Ibuprofen 400 mg: Every 8 hours if no contraindications\n- Cold compress",
            "Lifestyle": "Fluids (2-3 liters/day), rest, monitor temperature.",
            "Red Flags": "Fever >39°C for >3 days, severe headache, rash, breathing difficulty, altered consciousness.",
            "Follow-Up": "Review in 48 hours if fever persists."
        })
    
    # Diabetes
    elif re.search(r"\bdiabetes\b|\bhigh.*sugar\b|\bhyperglycemi\b", s):
        rec.update({
            "Diagnosis": "Type 2 Diabetes Mellitus",
            "Medicine": "- Metformin 500 mg: BD with meals (start low, titrate up)\n- Monitor blood glucose regularly",
            "Alternative": "- Glimepiride 1-2 mg OD (if metformin not tolerated)\n- DPP-4 inhibitors (Sitagliptin 100 mg OD)",
            "Lifestyle": "Low glycemic index diet, 150 min exercise/week, weight loss (if BMI >25), avoid refined sugars.",
            "Red Flags": "Glucose >400 mg/dL, confusion, chest pain, excessive thirst, fruity breath odor, rapid breathing.",
            "Follow-Up": "HbA1c every 3 months, annual eye/foot examination."
        })
    
    # Cold/URTI
    elif re.search(r"\bcough\b|\bcold\b|\bsneez\b|\brunny.*nose\b|\bnasal.*congest\b", s):
        rec.update({
            "Diagnosis": "Upper Respiratory Tract Infection (URTI)",
            "Medicine": "- Cetirizine 10 mg: Once daily at bedtime\n- Dextromethorphan cough syrup: 10 mL TDS\n- Saline nasal drops",
            "Alternative": "- Loratadine 10 mg OD (non-drowsy)\n- Steam inhalation 2-3 times daily\n- Honey (1 tsp) for cough",
            "Lifestyle": "Rest, warm fluids (tea, soup), avoid cold beverages, humidify room air.",
            "Red Flags": "High fever >38.5°C, chest pain, difficulty breathing, persistent symptoms >7 days.",
            "Follow-Up": "Review if symptoms persist beyond 5-7 days or worsen."
        })
    
    # Headache/Migraine
    elif re.search(r"\bheadache\b|\bmigrain\b|\bhead.*pain\b", s):
        rec.update({
            "Diagnosis": "Tension headache / Migraine",
            "Medicine": "- Paracetamol 500 mg: Every 6-8 hours (max 4g/day)\n- For migraine: Sumatriptan 50 mg as needed",
            "Alternative": "- Ibuprofen 400 mg TDS\n- Naproxen 250 mg BD\n- Rest in dark, quiet room",
            "Lifestyle": "Stress management, regular sleep (7-8 hrs), hydration, avoid triggers (caffeine, alcohol, screens).",
            "Red Flags": "Sudden severe headache (thunderclap), vision changes, confusion, neck stiffness, fever with headache.",
            "Follow-Up": "Review if headaches increase in frequency or severity. Consider CT if red flags present."
        })
    
    # Hypertension
    elif re.search(r"\bhypertension\b|\bhigh.*blood.*pressure\b|\bhbp\b", s):
        rec.update({
            "Diagnosis": "Essential Hypertension",
            "Medicine": "- Amlodipine 5 mg: Once daily\n- Monitor BP regularly (home monitoring)",
            "Alternative": "- Losartan 50 mg OD (ARB)\n- Enalapril 5 mg OD (ACE inhibitor)\n- Hydrochlorothiazide 12.5 mg OD",
            "Lifestyle": "Low sodium diet (<2g/day), DASH diet, regular exercise (30 min/day), weight reduction, limit alcohol, quit smoking.",
            "Red Flags": "BP >180/120, chest pain, severe headache, vision changes, shortness of breath, nosebleeds.",
            "Follow-Up": "BP monitoring weekly initially, then monthly. Review medications every 3 months."
        })
    
    # Asthma/Breathing problems
    elif re.search(r"\basthma\b|\bwheezing\b|\bshortness.*breath\b|\bbreathe\b", s):
        rec.update({
            "Diagnosis": "Asthma / Reactive Airway Disease",
            "Medicine": "- Salbutamol inhaler (2 puffs): PRN for symptoms\n- Budesonide inhaler 200 mcg: BD (controller)",
            "Alternative": "- Montelukast 10 mg: Once daily at bedtime\n- Formoterol + Budesonide combination inhaler",
            "Lifestyle": "Avoid triggers (dust, smoke, cold air), breathing exercises, maintain healthy weight, flu vaccination.",
            "Red Flags": "Severe difficulty breathing, blue lips/fingers, unable to speak full sentences, chest tightness not relieved by inhaler.",
            "Follow-Up": "Review in 2 weeks, peak flow monitoring, pulmonary function tests if persistent."
        })
    
    # Gastritis/Acid reflux
    elif re.search(r"\bgastric\b|\bacid\b|\bheart.*burn\b|\bindigestion\b|\bstomach.*pain\b|\bepigastric\b", s):
        rec.update({
            "Diagnosis": "Gastritis / Gastroesophageal Reflux Disease (GERD)",
            "Medicine": "- Omeprazole 20 mg: Once daily before breakfast\n- Antacid (Magaldrate) syrup: 10 mL after meals",
            "Alternative": "- Pantoprazole 40 mg OD\n- Ranitidine 150 mg BD\n- Sucralfate 1g QID",
            "Lifestyle": "Small frequent meals, avoid spicy/fatty foods, no late meals (3 hrs before bed), elevate head while sleeping, avoid alcohol/smoking.",
            "Red Flags": "Severe abdominal pain, vomiting blood, black tarry stools, weight loss, difficulty swallowing.",
            "Follow-Up": "Review in 4 weeks. Consider endoscopy if symptoms persist or red flags present."
        })
    
    # Allergic reactions
    elif re.search(r"\ballerg\b|\brash\b|\bitch\b|\bhives\b|\burticaria\b", s):
        rec.update({
            "Diagnosis": "Allergic Reaction / Urticaria",
            "Medicine": "- Cetirizine 10 mg: Once daily\n- Hydrocortisone cream 1%: Apply BD to affected areas\n- Avoid known allergens",
            "Alternative": "- Loratadine 10 mg OD\n- Fexofenadine 120 mg OD (non-sedating)\n- Calamine lotion for local relief",
            "Lifestyle": "Identify and avoid triggers, wear loose cotton clothing, avoid hot showers, keep skin moisturized.",
            "Red Flags": "Difficulty breathing, swelling of face/throat/tongue, rapid pulse, dizziness, loss of consciousness (anaphylaxis).",
            "Follow-Up": "Review in 1 week. Allergy testing if recurrent. Carry epinephrine auto-injector if severe allergies."
        })
    
    # Arthritis/Joint pain
    elif re.search(r"\barthritis\b|\bjoint.*pain\b|\bknee.*pain\b|\bback.*pain\b|\bosteo\b", s):
        rec.update({
            "Diagnosis": "Osteoarthritis / Degenerative Joint Disease",
            "Medicine": "- Ibuprofen 400 mg: TDS after meals\n- Glucosamine 1500 mg + Chondroitin 1200 mg: Once daily\n- Topical diclofenac gel",
            "Alternative": "- Naproxen 250 mg BD\n- Paracetamol 1g TDS\n- Hot/cold therapy\n- Capsaicin cream 0.025%",
            "Lifestyle": "Weight reduction if overweight, low-impact exercises (swimming, cycling), physical therapy, avoid prolonged standing.",
            "Red Flags": "Severe pain, joint swelling/warmth/redness, fever, inability to bear weight, deformity.",
            "Follow-Up": "Review in 2 weeks. X-rays if severe. Consider physiotherapy referral."
        })
    
    # Anxiety/Depression
    elif re.search(r"\banxiety\b|\bdepression\b|\bstress\b|\bpanic\b|\bmental\b|\bsad\b|\bworr\b", s):
        rec.update({
            "Diagnosis": "Anxiety / Depression - Requires Mental Health Evaluation",
            "Medicine": "- Escitalopram 10 mg: Once daily (after psychiatric evaluation)\n- Consider counseling/psychotherapy first",
            "Alternative": "- Sertraline 50 mg OD\n- Cognitive Behavioral Therapy (CBT)\n- Mindfulness-based therapy",
            "Lifestyle": "Regular exercise (30 min/day), adequate sleep (7-9 hrs), social support, relaxation techniques (meditation, yoga), limit caffeine/alcohol.",
            "Red Flags": "Suicidal thoughts, self-harm, severe panic attacks, inability to perform daily activities, hallucinations.",
            "Follow-Up": "Psychiatric referral recommended. Review in 1 week initially, then every 2-4 weeks."
        })
    
    # Urinary Tract Infection
    elif re.search(r"\buti\b|\burinary\b|\bburn.*urin\b|\bfrequent.*urin\b|\bdysuria\b", s):
        rec.update({
            "Diagnosis": "Urinary Tract Infection (UTI)",
            "Medicine": "- Nitrofurantoin 100 mg: BD for 5 days\n- Increase fluid intake (2-3 liters/day)",
            "Alternative": "- Trimethoprim 200 mg BD for 3 days\n- Ciprofloxacin 500 mg BD for 3 days\n- Cranberry supplements",
            "Lifestyle": "Hydration (8-10 glasses water/day), urinate frequently, avoid holding urine, proper hygiene, cranberry juice.",
            "Red Flags": "High fever, flank pain, blood in urine, nausea/vomiting, confusion (especially in elderly).",
            "Follow-Up": "Review if symptoms persist after 48 hours. Urine culture if recurrent UTIs."
        })
    
    # Thyroid disorders
    elif re.search(r"\bthyroid\b|\bhypothyroid\b|\bhyperthyroid\b|\bfatigue\b|\bweight.*gain\b", s):
        rec.update({
            "Diagnosis": "Thyroid Disorder (Requires lab confirmation)",
            "Medicine": "- Levothyroxine 50 mcg: Once daily (for hypothyroidism, after TSH confirmation)\n- Take on empty stomach",
            "Alternative": "- Dosage adjustment based on TSH levels\n- Regular monitoring required",
            "Lifestyle": "Regular medication timing, avoid soy/calcium supplements near medication time, balanced diet, regular exercise.",
            "Red Flags": "Severe fatigue, rapid heart rate, tremors, significant weight changes, neck swelling.",
            "Follow-Up": "TSH levels every 6-8 weeks initially, then every 6 months once stable."
        })
    
    # Skin infections
    elif re.search(r"\bskin.*infection\b|\bfungal\b|\bringworm\b|\beczema\b|\bdermatitis\b", s):
        rec.update({
            "Diagnosis": "Skin Infection / Dermatitis",
            "Medicine": "- Clotrimazole cream 1%: Apply BD for fungal infections\n- Hydrocortisone cream 1%: BD for inflammation (max 7 days)",
            "Alternative": "- Terbinafine cream 1% BD\n- Mupirocin ointment (if bacterial)\n- Calamine lotion for soothing",
            "Lifestyle": "Keep area clean and dry, avoid tight clothing, change clothes daily, avoid sharing towels.",
            "Red Flags": "Spreading infection, fever, pus discharge, severe pain, no improvement in 1 week.",
            "Follow-Up": "Review in 1 week if no improvement. Skin scraping/culture if persistent."
        })
    
    # Anemia
    elif re.search(r"\banemia\b|\banemic\b|\blow.*iron\b|\bfatigue\b|\bpale\b|\bdizz\b", s):
        rec.update({
            "Diagnosis": "Iron Deficiency Anemia (Requires lab confirmation)",
            "Medicine": "- Ferrous sulfate 325 mg: Once daily with vitamin C\n- Take on empty stomach or with orange juice",
            "Alternative": "- Ferrous gluconate 300 mg OD (if GI side effects)\n- Iron polymaltose complex\n- Vitamin B12 if deficient",
            "Lifestyle": "Iron-rich foods (red meat, spinach, lentils, fortified cereals), vitamin C with meals (enhances absorption), avoid tea/coffee with meals.",
            "Red Flags": "Severe fatigue, chest pain, shortness of breath, rapid heartbeat, severe dizziness, blood in stool.",
            "Follow-Up": "Hemoglobin check in 4-6 weeks. Continue iron for 3-6 months to replenish stores."
        })
    
    # Insomnia/Sleep disorders
    elif re.search(r"\binsomnia\b|\bsleep\b|\bcan't.*sleep\b|\bawake\b", s):
        rec.update({
            "Diagnosis": "Insomnia / Sleep Disorder",
            "Medicine": "- Melatonin 3 mg: 30 minutes before bedtime\n- Short-term: Zolpidem 5 mg (if severe, max 2 weeks)",
            "Alternative": "- Diphenhydramine 25 mg at bedtime\n- Trazodone 50 mg (if depression present)\n- CBT for insomnia (CBT-I)",
            "Lifestyle": "Sleep hygiene: regular sleep schedule, dark/cool room, avoid screens 1 hr before bed, no caffeine after 2 PM, relaxation techniques.",
            "Red Flags": "Sleep apnea symptoms (snoring, gasping), severe daytime impairment, depression with insomnia.",
            "Follow-Up": "Review in 2 weeks. Sleep study if suspected sleep apnea."
        })
    
    return rec

def save_visit(patient_id, doctor_id, data, recommendation):
    """Save visit to database"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Add doctor-patient mapping
    add_doctor_patient_mapping(doctor_id, patient_id)
    
    cur.execute("""
        INSERT INTO visit (patient_id, doctor_id, symptoms, age, gender, genetic_history,
                          medicine, diagnosis, lifestyle, follow_up, ml_prediction, ml_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id, doctor_id, data.get('symptoms'), data.get('age'), data.get('gender'),
        data.get('genetic_history'), recommendation.get('Medicine'), recommendation.get('Diagnosis'),
        recommendation.get('Lifestyle'), recommendation.get('Follow-Up'),
        recommendation.get('ml_prediction'), recommendation.get('ml_confidence')
    ))
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Main app logic
def main():
    # Login/Register page
    if not st.session_state.authenticated:
        st.title("🩺 Med4Me")
        st.subheader("Clinical Decision Support System")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                
                if submit:
                    user_id = authenticate_user(username, password)
                    if user_id:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_id = user_id
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
            
            st.info("Demo Account: **admin** / **admin123**")
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                register = st.form_submit_button("Register")
                
                if register:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        user_id, error = register_user(new_username, new_password)
                        if user_id:
                            st.success("Registration successful! Please login.")
                        else:
                            st.error(error or "Registration failed")
        
        return
    
    # Main application
    st.title(f"🩺 Med4Me - Doctor Chat")
    st.caption(f"Logged in as **{st.session_state.username}**")
    
    # Sidebar - Patient list
    with st.sidebar:
        st.header("My Patients")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("➕ New", use_container_width=True):
                st.session_state.current_patient = None
                st.session_state.patient_data = {}
                st.session_state.step = 0
                st.rerun()
        
        st.divider()
        
        patients = get_doctor_patients(st.session_state.user_id)
        
        if patients:
            for patient in patients:
                patient_id, created_at, visit_count, last_visit, last_symptoms = patient
                
                is_selected = st.session_state.current_patient == patient_id
                
                if st.button(
                    f"**{patient_id}**\nVisits: {visit_count}\nLast: {last_visit[:10] if last_visit else 'N/A'}",
                    key=patient_id,
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    st.session_state.current_patient = patient_id
                    st.session_state.step = 0
                    st.rerun()
        else:
            st.info("No patients yet")
        
        st.divider()
        st.caption("Your patients are scoped to your account")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        if USE_ML:
            st.success("✅ ML Model Active")
        else:
            st.warning("⚠️ Rule-based System")
    
    # Main content area
    if st.session_state.current_patient:
        # Show patient history
        st.subheader(f"Patient: {st.session_state.current_patient}")
        
        history = get_patient_history(st.session_state.current_patient, st.session_state.user_id)
        
        if history:
            with st.expander("📋 Past Visits", expanded=False):
                for visit in history:
                    st.markdown(f"""
                    **Visit Date:** {visit[3]}  
                    **Symptoms:** {visit[4]}  
                    **Diagnosis:** {visit[9]}  
                    **Medicine:** {visit[8][:100]}...
                    """)
                    st.divider()
            
            # Pre-fill data from last visit
            last_visit = history[-1]
            st.session_state.patient_data = {
                'patient_id': st.session_state.current_patient,
                'age': last_visit[5],
                'gender': last_visit[6],
                'genetic_history': last_visit[7] or "Not provided"
            }
            
            st.info(f"Age: {last_visit[5]} | Gender: {last_visit[6]} | Genetic History: {last_visit[7] or 'Not provided'}")
            
            # Only ask for symptoms
            with st.form("symptoms_form"):
                symptoms = st.text_area("Enter current symptoms:", height=150)
                submit = st.form_submit_button("Generate Recommendation")
                
                if submit and symptoms:
                    st.session_state.patient_data['symptoms'] = symptoms
                    
                    with st.spinner("Generating recommendation..."):
                        recommendation = ml_recommendation(
                            symptoms,
                            st.session_state.patient_data['age'],
                            st.session_state.patient_data['gender'],
                            st.session_state.patient_data['genetic_history']
                        )
                        
                        save_visit(
                            st.session_state.current_patient,
                            st.session_state.user_id,
                            st.session_state.patient_data,
                            recommendation
                        )
                        
                        show_recommendation(recommendation)
        else:
            st.info("No past history. Creating new patient record.")
            # New patient workflow
            new_patient_form()
    else:
        # New patient workflow
        st.subheader("New Patient Consultation")
        new_patient_form()

def new_patient_form():
    """Form for new patient consultation"""
    with st.form("new_patient_form"):
        patient_id = st.text_input("Patient ID (format: P12345)")
        age = st.number_input("Age", min_value=1, max_value=150, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        genetic_history = st.text_area("Genetic/Family History", 
                                       placeholder="e.g., Family history of diabetes, hypertension...")
        symptoms = st.text_area("Current Symptoms", height=150)
        
        submit = st.form_submit_button("Generate Recommendation")
        
        if submit:
            # Validate patient ID
            if not re.match(r'^P\d+$', patient_id):
                st.error("Patient ID must be in format P12345")
                return
            
            if not symptoms or len(symptoms) < 3:
                st.error("Please enter valid symptoms")
                return
            
            data = {
                'patient_id': patient_id,
                'age': str(age),
                'gender': gender.lower(),
                'genetic_history': genetic_history or "None",
                'symptoms': symptoms
            }
            
            with st.spinner("Generating recommendation..."):
                recommendation = ml_recommendation(symptoms, age, gender.lower(), genetic_history)
                
                save_visit(patient_id, st.session_state.user_id, data, recommendation)
                
                st.session_state.current_patient = patient_id
                show_recommendation(recommendation)

def show_recommendation(rec):
    """Display medical recommendation"""
    st.success("✅ Recommendation Generated")
    
    st.markdown("### 📋 Medical Recommendation")
    
    st.markdown(f"""
    <div class="rec-section">
    <strong>🔍 Diagnosis:</strong><br/>
    {rec.get('Diagnosis', 'N/A')}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="rec-section">
    <strong>💊 Medication:</strong><br/>
    {rec.get('Medicine', 'N/A').replace('- ', '<br/>- ')}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="rec-section">
    <strong>🔄 Alternative Medication:</strong><br/>
    {rec.get('Alternative', 'N/A').replace('- ', '<br/>- ')}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="rec-section">
    <strong>🏃 Lifestyle Recommendations:</strong><br/>
    {rec.get('Lifestyle', 'N/A')}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="rec-section warning">
    <strong>⚠️ Red Flags / Warning Signs:</strong><br/>
    {rec.get('Red Flags', 'N/A')}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="rec-section">
    <strong>📅 Follow-Up:</strong><br/>
    {rec.get('Follow-Up', 'N/A')}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="rec-section">
    <strong>📝 Notes:</strong><br/>
    <em>{rec.get('Notes', 'N/A')}</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

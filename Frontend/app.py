from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to the pretrained models

MODEL_PATHS = {
    "Logistic Regression": '../models/Logistic_Regression.pkl',
    "SVM": '../models/SVM.pkl',
    "Decision Tree": '../models/Decision_Tree.pkl',
    "Random Forest": '../models/Random_Forest.pkl',
    "Gradient Boosting": '../models/Gradient_Boosting.pkl',
    "XGBoost": '../models/XGBoost.pkl',
    "CatBoost": '../models/CatBoost.pkl',
    "K-Nearest Neighbors": '../models/K-Nearest_Neighbors.pkl',
    "Naive Bayes": '../models/Naive_Bayes.pkl'
}


# Initialize dictionaries to store models and scalers
models = {}
scalers = {}

def load_models():
    """Load all models and scalers from files"""
    try:
        for model_name, path in MODEL_PATHS.items():
            if os.path.exists(path):
                with open(path, 'rb') as file:
                    models[model_name] = pickle.load(file)
                
                # Load corresponding scaler if it exists
                scaler_path = f'model/scaler_{model_name.lower().replace(" ", "_")}.pkl'
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as file:
                        scalers[model_name] = pickle.load(file)
                
                logger.info(f"Successfully loaded {model_name}")
            else:
                logger.warning(f"Model file not found: {path}")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# Feature names in the correct order
FEATURE_NAMES = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
    'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn',
    'dm', 'cad', 'appet', 'pe', 'ane'
]

def preprocess_features(form_data):
    """Convert form data into numpy array with proper feature ordering"""
    try:
        features = []
        for feature in FEATURE_NAMES:
            value = form_data.get(feature)
            if value is None:
                raise ValueError(f"Missing feature: {feature}")
            
            # Handle categorical variables
            if feature in ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']:
                if feature in ['rbc', 'pc']:
                    features.append(1 if value.lower() == 'normal' else 0)
                elif feature in ['pcc', 'ba']:
                    features.append(1 if value.lower() == 'present' else 0)
                elif feature == 'appet':
                    features.append(1 if value.lower() == 'good' else 0)
                else:
                    features.append(1 if value.lower() == 'yes' else 0)
            else:
                features.append(float(value))
        
        return np.array(features).reshape(1, -1)
    except Exception as e:
        logger.error(f"Error in preprocessing features: {str(e)}")
        raise
@app.route('/')
def dashboard():
    """Render the dashboard page"""
    return render_template('dashboard.html')

@app.route('/test_ckd')
def test_ckd():
    """Render the CKD test form"""
    return render_template('test_ckd.html')
# @app.route('/')
# def home():
#     """Render the home page"""
#     return render_template('test_ckd.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests and return results from all models"""
    try:
        # Extract and preprocess features
        form_data = request.form
        features = preprocess_features(form_data)
        
        # Store all model predictions
        all_predictions = []
        
        # Make predictions using all models
        for model_name, model in models.items():
            # Scale features if scaler exists for this model
            if model_name in scalers:
                scaled_features = scalers[model_name].transform(features)
            else:
                scaled_features = features
            
            # Get prediction and probability scores
            prediction = model.predict(scaled_features)[0]
            
            try:
                probability = model.predict_proba(scaled_features)[0]
                confidence = round(max(probability) * 100, 2)  # Convert to percentage
            except:
                confidence = None
            
            all_predictions.append({
                'model_name': model_name,
                'prediction': 1 if prediction == 1 else 0,
                'result': "Chronic Kidney Disease (CKD)" if prediction == 1 else "No CKD Detected",
                'confidence': confidence,
                'status': 'danger' if prediction == 1 else 'safe'
            })
        
        # Calculate ensemble prediction (majority voting)
        ckd_votes = sum(1 for p in all_predictions if p['prediction'] == 1)
        ensemble_result = {
            'prediction': 1 if ckd_votes > len(all_predictions)/2 else 0,
            'result': "Chronic Kidney Disease (CKD)" if ckd_votes > len(all_predictions)/2 else "No CKD Detected",
            'confidence': round((max(ckd_votes, len(all_predictions) - ckd_votes) / len(all_predictions)) * 100, 2),
            'status': 'danger' if ckd_votes > len(all_predictions)/2 else 'safe'
        }
        
        # Get input feature values for display
        feature_values = {
            'Age': form_data.get('age'),
            'Blood Pressure': form_data.get('bp'),
            'Specific Gravity': form_data.get('sg'),
            'Albumin': form_data.get('al'),
            'Sugar': form_data.get('su'),
            'RBC': form_data.get('rbc'),
            'Blood Glucose': form_data.get('bgr'),
            'Blood Urea': form_data.get('bu'),
            'Serum Creatinine': form_data.get('sc'),
            'Sodium': form_data.get('sod'),
            'Potassium': form_data.get('pot'),
            'Hemoglobin': form_data.get('hemo'),
            'Packed Cell Volume': form_data.get('pcv'),
            'White Blood Cells': form_data.get('wc'),
            'Red Blood Cells': form_data.get('rc'),
            'Hypertension': form_data.get('htn'),
            'Diabetes Mellitus': form_data.get('dm'),
            'Coronary Artery Disease': form_data.get('cad'),
            'Appetite': form_data.get('appet'),
            'Pedal Edema': form_data.get('pe'),
            'Anemia': form_data.get('ane')
        }
        
        return render_template(
            'result.html',
            predictions=all_predictions,
            ensemble_result=ensemble_result,
            feature_values=feature_values
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('error.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        features = preprocess_features(request.json)
        predictions = {}
        
        for model_name, model in models.items():
            if model_name in scalers:
                scaled_features = scalers[model_name].transform(features)
            else:
                scaled_features = features
                
            prediction = model.predict(scaled_features)[0]
            try:
                probability = model.predict_proba(scaled_features)[0]
                confidence = max(probability) * 100
            except:
                confidence = None
                
            predictions[model_name] = {
                'prediction': int(prediction),
                'confidence': round(confidence, 2) if confidence is not None else None,
                'label': "CKD" if prediction == 1 else "Not CKD"
            }
        
        return jsonify({
            'status': 'success',
            'predictions': predictions
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

# Load models when the application starts
load_models()

if __name__ == '__main__':
    app.run(debug=True)







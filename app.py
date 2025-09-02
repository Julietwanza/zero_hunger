from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import requests
from datetime import datetime, timedelta
import sqlite3

app = Flask(__name__)
CORS(app)

# Configuration
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', 'demo_key')
DATABASE = 'agriculture.db'

# Initialize database
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            location TEXT,
            crop_type TEXT,
            predicted_yield REAL,
            confidence REAL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_type TEXT,
            season TEXT,
            recommendation TEXT,
            priority INTEGER
        )
    ''')
    conn.commit()
    conn.close()

# Kenyan crop data and ML model
class CropPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.is_trained = False
        self.train_model()
    
    def train_model(self):
        # Sample training data for Kenyan agriculture
        training_data = {
            'crop_type': ['maize', 'beans', 'wheat', 'rice', 'sorghum'] * 100,
            'season': ['long_rains', 'short_rains', 'dry'] * 166 + ['long_rains', 'short_rains'],
            'rainfall_mm': np.random.normal(800, 200, 500),
            'temperature_c': np.random.normal(22, 5, 500),
            'soil_ph': np.random.normal(6.5, 0.5, 500),
            'fertilizer_kg_per_ha': np.random.normal(50, 15, 500),
            'yield_tons_per_ha': np.random.normal(2.5, 0.8, 500)
        }
        
        df = pd.DataFrame(training_data)
        
        # Encode categorical variables
        categorical_cols = ['crop_type', 'season']
        for col in categorical_cols:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Prepare features and target
        feature_cols = ['crop_type_encoded', 'season_encoded', 'rainfall_mm', 
                       'temperature_c', 'soil_ph', 'fertilizer_kg_per_ha']
        X = df[feature_cols]
        y = df['yield_tons_per_ha']
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict_yield(self, crop_type, season, weather_data, soil_ph=6.5, fertilizer_kg=50):
        if not self.is_trained:
            return None
        
        # Encode inputs
        try:
            crop_encoded = self.label_encoders['crop_type'].transform([crop_type])[0]
            season_encoded = self.label_encoders['season'].transform([season])[0]
        except:
            # Handle unknown categories
            crop_encoded = 0
            season_encoded = 0
        
        # Prepare features
        features = [[
            crop_encoded,
            season_encoded,
            weather_data.get('rainfall', 800),
            weather_data.get('temperature', 22),
            soil_ph,
            fertilizer_kg
        ]]
        
        prediction = self.model.predict(features)[0]
        
        # Calculate confidence (simplified)
        confidence = min(0.95, max(0.6, 0.85 - abs(prediction - 2.5) * 0.1))
        
        return {
            'predicted_yield': round(prediction, 2),
            'confidence': round(confidence, 2),
            'unit': 'tons per hectare'
        }

# Initialize services
predictor = CropPredictor()
init_db()

# Weather service
def get_weather_data(location):
    """Fetch weather data from OpenWeatherMap API"""
    if OPENWEATHER_API_KEY == 'demo_key':
        # Return mock data for demo
        return {
            'temperature': 24.5,
            'humidity': 65,
            'rainfall': 45.2,
            'wind_speed': 3.2,
            'description': 'Partly cloudy',
            'location': location
        }
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': f"{location},KE",
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'rainfall': data.get('rain', {}).get('1h', 0),
            'wind_speed': data['wind']['speed'],
            'description': data['weather'][0]['description'],
            'location': location
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        return {
            'temperature': 22,
            'humidity': 60,
            'rainfall': 30,
            'wind_speed': 2.5,
            'description': 'Data unavailable',
            'location': location
        }

# Farming recommendations
FARMING_RECOMMENDATIONS = {
    'maize': {
        'long_rains': [
            "Plant maize seeds at the onset of long rains (March-April)",
            "Use certified seeds like H516, H614, or local varieties",
            "Apply 50kg DAP fertilizer per hectare at planting",
            "Top-dress with 50kg CAN fertilizer 6-8 weeks after planting"
        ],
        'short_rains': [
            "Choose short-season maize varieties for October planting",
            "Ensure proper land preparation before rains",
            "Consider drought-tolerant varieties in arid areas"
        ]
    },
    'beans': {
        'long_rains': [
            "Plant beans varieties like Rose coco, Mwezi moja, Canadian wonder",
            "Space rows 30-45cm apart with 10cm between plants",
            "Apply organic manure or compost before planting",
            "Practice crop rotation to improve soil fertility"
        ],
        'short_rains': [
            "Use early-maturing varieties for short rains season",
            "Ensure good drainage to prevent waterlogging",
            "Harvest when pods are mature but not over-dried"
        ]
    },
    'tea': {
        'year_round': [
            "Prune tea bushes every 4-5 years for optimal yield",
            "Apply fertilizer (NPK 25:5:5) three times per year",
            "Maintain proper plucking intervals (7-14 days)",
            "Control weeds and pests regularly"
        ]
    },
    'coffee': {
        'year_round': [
            "Apply coffee fertilizer during rainy seasons",
            "Practice proper pruning and disease management",
            "Harvest only ripe cherries for better quality",
            "Process coffee within 24 hours of picking"
        ]
    }
}

# Routes
@app.route('/')
def home():
    return jsonify({
        'message': 'Kenya Zero Hunger AI Agriculture Assistant',
        'version': '1.0',
        'endpoints': [
            '/api/predict-yield',
            '/api/weather/<location>',
            '/api/recommendations',
            '/api/crops'
        ]
    })

@app.route('/api/predict-yield', methods=['POST'])
def predict_yield():
    try:
        data = request.get_json()
        
        crop_type = data.get('crop_type', 'maize').lower()
        location = data.get('location', 'Nairobi')
        season = data.get('season', 'long_rains')
        soil_ph = float(data.get('soil_ph', 6.5))
        fertilizer_kg = float(data.get('fertilizer_kg', 50))
        
        # Get weather data
        weather_data = get_weather_data(location)
        
        # Make prediction
        result = predictor.predict_yield(
            crop_type, season, weather_data, soil_ph, fertilizer_kg
        )
        
        if result:
            # Store prediction in database
            conn = sqlite3.connect(DATABASE)
            c = conn.cursor()
            c.execute('''
                INSERT INTO predictions (location, crop_type, predicted_yield, confidence)
                VALUES (?, ?, ?, ?)
            ''', (location, crop_type, result['predicted_yield'], result['confidence']))
            conn.commit()
            conn.close()
            
            return jsonify({
                'success': True,
                'prediction': result,
                'weather': weather_data,
                'inputs': {
                    'crop_type': crop_type,
                    'location': location,
                    'season': season,
                    'soil_ph': soil_ph,
                    'fertilizer_kg': fertilizer_kg
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Prediction failed'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/weather/<location>')
def get_weather(location):
    weather_data = get_weather_data(location)
    return jsonify(weather_data)

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        crop_type = data.get('crop_type', 'maize').lower()
        season = data.get('season', 'long_rains').lower()
        
        recommendations = FARMING_RECOMMENDATIONS.get(crop_type, {}).get(
            season, FARMING_RECOMMENDATIONS.get(crop_type, {}).get('year_round', [])
        )
        
        if not recommendations:
            recommendations = [
                f"General farming advice for {crop_type}:",
                "Ensure proper soil preparation and drainage",
                "Use quality seeds from certified dealers",
                "Apply appropriate fertilizers based on soil test",
                "Practice integrated pest management",
                "Harvest at the right maturity stage"
            ]
        
        return jsonify({
            'success': True,
            'crop_type': crop_type,
            'season': season,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/crops')
def get_supported_crops():
    crops = {
        'cereals': ['maize', 'wheat', 'rice', 'sorghum', 'millet'],
        'legumes': ['beans', 'cowpeas', 'green_grams', 'pigeon_peas'],
        'cash_crops': ['tea', 'coffee', 'cotton', 'sugarcane'],
        'vegetables': ['kale', 'tomatoes', 'onions', 'carrots', 'cabbage'],
        'fruits': ['mango', 'banana', 'avocado', 'passion_fruit']
    }
    
    return jsonify({
        'success': True,
        'crops': crops,
        'total_supported': sum(len(category) for category in crops.values())
    })

@app.route('/api/statistics')
def get_statistics():
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        # Get prediction statistics
        c.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = c.fetchone()[0]
        
        c.execute('''
            SELECT crop_type, COUNT(*) as count 
            FROM predictions 
            GROUP BY crop_type 
            ORDER BY count DESC 
            LIMIT 5
        ''')
        popular_crops = c.fetchall()
        
        c.execute('''
            SELECT AVG(predicted_yield) as avg_yield
            FROM predictions
            WHERE timestamp > datetime('now', '-30 days')
        ''')
        avg_yield = c.fetchone()[0] or 0
        
        conn.close()
        
        return jsonify({
            'success': True,
            'statistics': {
                'total_predictions': total_predictions,
                'popular_crops': [{'crop': crop, 'count': count} for crop, count in popular_crops],
                'average_yield_last_30_days': round(avg_yield, 2)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
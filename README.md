# Kenya Zero Hunger AI - Agriculture Assistant

An AI-powered web application that helps small agricultural businesses in Kenya optimize crop yields and reduce food waste through intelligent farming recommendations.

## 🌾 Features

- **Crop Yield Prediction** - ML-based predictions for major Kenyan crops
- **Weather Integration** - Real-time weather data for informed decisions  
- **Farming Recommendations** - Season-specific advice for different crops
- **Multi-language Support** - English and Swahili interface
- **Mobile Friendly** - Works on smartphones and tablets

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/kenya-zero-hunger-ai.git
cd kenya-zero-hunger-ai
```

### 2. Setup Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### 3. Setup Frontend
```bash
cd frontend
python -m http.server 8080
# Or simply open index.html in your browser
```

### 4. Access Application
- Frontend: http://localhost:8080
- Backend API: http://localhost:5000

## 📦 Requirements

**Backend Requirements** (requirements.txt):
```
Flask==2.3.2
Flask-CORS==4.0.0
pandas==2.0.3
scikit-learn==1.3.0
numpy==1.24.3
requests==2.31.0
joblib==1.3.1
```

**Frontend**: Pure HTML/CSS/JavaScript (no dependencies)

## 🌍 Supported Crops

- **Cereals**: Maize, Wheat, Rice, Sorghum
- **Legumes**: Beans, Cowpeas, Green grams
- **Cash Crops**: Tea, Coffee, Cotton
- **Vegetables**: Kale, Tomatoes, Onions

## 🔧 Configuration

### Environment Variables
```bash
export OPENWEATHER_API_KEY="your_api_key_here"
export FLASK_ENV="development"
```

### Get Weather API Key (Optional)
1. Sign up at [OpenWeatherMap](https://openweathermap.org/api)
2. Get your free API key
3. Set the environment variable

*Note: The app works with demo data if no API key is provided*

## 📱 Usage

1. **Select Crop**: Choose from supported Kenyan crops
2. **Enter Location**: Select your farming area
3. **Choose Season**: Pick the growing season
4. **Add Details**: Input soil pH and fertilizer amounts
5. **Get Prediction**: Receive yield predictions and recommendations

## 🚢 Deployment Options

### Free Hosting
- **Frontend**: GitHub Pages, Netlify
- **Backend**: Heroku, Railway, Render

### Deploy to Heroku
```bash
# Create Procfile in backend folder
echo "web: python app.py" > Procfile

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## 📧 Support

- **Issues**: Open a GitHub issue
- **Questions**: Create a discussion
- **Email**: your-email@example.com

## 📄 License

MIT License - Free for commercial and non-commercial use

---

**Built for SDG 2: Zero Hunger** 🎯  
Empowering Kenyan farmers with AI technology


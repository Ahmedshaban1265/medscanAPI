# Medical AI Flask API

AI-powered medical diagnosis API for brain tumors and skin cancer detection, deployed on Railway.

## 🏥 Features
- **Brain Tumor Classification**: Detect and classify brain tumors (Glioma, Meningioma, Pituitary tumor, No tumor)
- **Brain Tumor Segmentation**: Generate segmentation masks for brain tumors
- **Skin Cancer Detection**: Classify skin lesions across 7 different types

## 🚀 Live Demo
Deployed on Railway: `https://your-app-name.up.railway.app`

## 📋 API Endpoints

### Health Check
```
GET /health
```
Returns API status and health information.

### Models Status
```
GET /models/status
```
Check which AI models are loaded and available.

### Image Analysis
```
POST /scan
```
**Parameters:**
- `image`: Image file (JPG, PNG)
- `diseaseType`: Either "Brain Tumor" or "Skin Cancer"

**Example Response:**
```json
{
  "diagnosis": "No tumor",
  "confidence": "95.67%",
  "description": "Based on the AI analysis, no brain tumor was detected.",
  "recommendations": ["Routine check-up", "Maintain healthy lifestyle"]
}
```

## 🛠 Technology Stack
- **Backend**: Flask (Python)
- **AI/ML**: TensorFlow, Keras
- **Image Processing**: PIL, NumPy
- **Deployment**: Railway
- **Server**: Gunicorn

## 📁 Project Structure
```
medical-ai-railway/
├── app.py                              # Main Flask application
├── requirements.txt                    # Python dependencies
├── nixpacks.toml                      # Railway configuration
├── Procfile                           # Process configuration
├── .gitattributes                     # Git LFS for large files
├── Brain_Tumor_Classification_Model.h5
├── brain_tumor_segmentation_model.h5
├── skin_cancer_model.h5
└── README.md
```

## 🔧 Local Development

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/username/medical-ai-railway.git
cd medical-ai-railway

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The API will be available at `http://localhost:5000`

## 🚀 Deployment on Railway

### Method 1: Deploy from GitHub (Recommended)
1. Fork this repository
2. Go to [Railway](https://railway.app)
3. Click "Start a New Project"
4. Select "Deploy from GitHub repo"
5. Choose your forked repository
6. Railway will automatically detect and deploy your Flask app

### Method 2: Deploy using Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy
railway up
```

## 📊 Model Information

### Brain Tumor Classification
- **Input Size**: 128x128x3
- **Classes**: Glioma, Meningioma, Pituitary tumor, No tumor
- **Architecture**: CNN-based classification model

### Brain Tumor Segmentation
- **Input Size**: 128x128x3
- **Output**: Segmentation mask
- **Purpose**: Identify tumor regions in brain scans

### Skin Cancer Detection
- **Input Size**: 28x28x3
- **Classes**: 7 types including Melanoma, Basal cell carcinoma, etc.
- **Dataset**: Based on HAM10000 dataset

## ⚠️ Important Notes

### Medical Disclaimer
This AI system is for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical decisions.

### Performance Optimization
- Models are loaded on-demand to optimize memory usage
- Supports concurrent requests with thread-safe model loading
- Optimized for Railway's 1GB memory limit

## 🔍 Testing the API

### Using cURL
```bash
# Health check
curl https://your-app-name.up.railway.app/health

# Upload image for brain tumor analysis
curl -X POST https://your-app-name.up.railway.app/scan \
  -F "image=@brain_scan.jpg" \
  -F "diseaseType=Brain Tumor"

# Upload image for skin cancer analysis
curl -X POST https://your-app-name.up.railway.app/scan \
  -F "image=@skin_lesion.jpg" \
  -F "diseaseType=Skin Cancer"
```

### Using Python
```python
import requests

# Health check
response = requests.get('https://your-app-name.up.railway.app/health')
print(response.json())

# Image analysis
files = {'image': open('test_image.jpg', 'rb')}
data = {'diseaseType': 'Brain Tumor'}
response = requests.post('https://your-app-name.up.railway.app/scan', 
                        files=files, data=data)
print(response.json())
```

## 📈 Monitoring and Logs

### Railway Dashboard
- View real-time logs in Railway dashboard
- Monitor memory and CPU usage
- Track deployment history

### Log Levels
The application logs important events:
- ✅ Model loading success
- ⚠️ Model file not found warnings
- ❌ Error messages with details

## 💰 Cost and Limits

### Railway Free Tier
- **500 hours/month** of runtime
- **1GB RAM**
- **1GB storage**
- **100GB bandwidth/month**

### Optimization Tips
- Models load on-demand to save memory
- Efficient image preprocessing
- Proper error handling to prevent crashes

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support
If you encounter any issues:
1. Check the Railway logs for error messages
2. Verify your model files are properly uploaded
3. Ensure image format is supported (JPG, PNG)
4. Contact support if problems persist

## 🔗 Links
- [Railway Documentation](https://docs.railway.com)
- [Flask Documentation](https://flask.palletsprojects.com)
- [TensorFlow Documentation](https://tensorflow.org)


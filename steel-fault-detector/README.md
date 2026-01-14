# 🏭 Steel Plate Fault Detector

A modern web application that uses **Convolutional Neural Networks (CNN)** to identify and classify defects in steel plates from images.

## 📋 Project Overview

This project is part of a **Material Science** course that demonstrates the application of Machine Learning in industrial quality control. The system can identify **7 types of faults** commonly found in steel plates:

1. **Pastry** - Flaky or layered surface defects
2. **Z_Scratch** - Z-shaped or zigzag scratch patterns
3. **K_Scratch** - K-shaped scratch marks
4. **Stains** - Discoloration or chemical spots
5. **Dirtiness** - Surface contamination
6. **Bumps** - Raised areas or protrusions
7. **Other_Faults** - Miscellaneous defects

## 🛠️ Tech Stack

### Backend
- **Python 3.9+**
- **Flask** - Web framework
- **TensorFlow/Keras** - Deep Learning
- **OpenCV & Pillow** - Image processing

### Frontend
- **React 18** - UI framework
- **CSS3** - Modern styling with gradients and animations
- **Axios** - API communication
- **Framer Motion** - Smooth animations
- **React Dropzone** - File upload handling

## 📁 Project Structure

```
steel-fault-detector/
├── backend/
│   ├── app.py              # Flask API server
│   ├── train_model.py      # CNN model training script
│   ├── requirements.txt    # Python dependencies
│   └── models/             # Trained model storage
│       └── .gitkeep
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   ├── App.css         # Component styles
│   │   ├── index.js        # Entry point
│   │   └── index.css       # Global styles
│   └── package.json
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Node.js 16 or higher
- npm or yarn

### Backend Setup

1. **Navigate to the backend directory:**
   ```powershell
   cd steel-fault-detector/backend
   ```

2. **Create a virtual environment:**
   ```powershell
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   ```powershell
   # Windows
   .\venv\Scripts\Activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

5. **Run the Flask server:**
   ```powershell
   python app.py
   ```
   The API will be available at `http://localhost:5000`

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```powershell
   cd steel-fault-detector/frontend
   ```

2. **Install dependencies:**
   ```powershell
   npm install
   ```

3. **Start the development server:**
   ```powershell
   npm start
   ```
   The app will open at `http://localhost:3000`

## 🧠 Training the CNN Model

To train the model with your own data:

1. **Organize training images:**
   ```
   backend/data/train/
   ├── Pastry/
   │   ├── image1.jpg
   │   └── ...
   ├── Z_Scratch/
   ├── K_Scratch/
   ├── Stains/
   ├── Dirtiness/
   ├── Bumps/
   └── Other_Faults/
   ```

2. **Run the training script:**
   ```powershell
   cd backend
   python train_model.py
   ```

3. The trained model will be saved to `backend/models/steel_fault_cnn.h5`

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check API status |
| `/api/fault-types` | GET | Get all fault types with descriptions |
| `/api/predict` | POST | Analyze an uploaded image |
| `/api/demo-predict` | POST | Demo prediction (works without trained model) |

## 🎨 Features

- ✅ **Drag & Drop** image upload
- ✅ **Real-time** fault detection
- ✅ **Confidence scores** for all fault types
- ✅ **Beautiful dark theme** UI
- ✅ **Responsive design** for all devices
- ✅ **Demo mode** when model is not trained
- ✅ **Animated** results display

## 📸 Screenshots

The application features:
- Modern dark industrial theme
- Gradient accents and glass-morphism effects
- Smooth animations and transitions
- Progress bars for confidence visualization

## 🔬 CNN Model Architecture

The model uses a custom CNN architecture with:
- 5 Convolutional blocks with BatchNormalization
- MaxPooling and Dropout for regularization
- GlobalAveragePooling for feature extraction
- Dense layers for classification
- Softmax output for 7 fault classes

Alternative: Transfer Learning with MobileNetV2 for better accuracy with limited data.

## 📚 Course Information

- **Subject:** Material Science
- **Project:** ML Model to Identify Type of Fault in Steel Plate
- **Semester:** 3rd Semester
- **Year:** 2024-2028

## 📝 License

This project is for educational purposes as part of academic coursework.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements!

---

Made with ❤️ for Material Science Project

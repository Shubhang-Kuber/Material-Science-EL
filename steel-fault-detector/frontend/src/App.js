import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FaIndustry, 
  FaCloudUploadAlt, 
  FaSearch, 
  FaTimes, 
  FaCheckCircle,
  FaExclamationTriangle,
  FaInfoCircle,
  FaBug,
  FaWater,
  FaCircle,
  FaMountain,
  FaQuestion
} from 'react-icons/fa';
import axios from 'axios';
import './App.css';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Fault type icons mapping
const faultIcons = {
  'Pastry': FaCircle,
  'Z_Scratch': FaBug,
  'K_Scratch': FaBug,
  'Stains': FaWater,
  'Dirtiness': FaWater,
  'Bumps': FaMountain,
  'Other_Faults': FaQuestion
};

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState({ healthy: false, modelLoaded: false });
  const [faultTypes, setFaultTypes] = useState([]);

  // Check API health on mount
  useEffect(() => {
    checkApiHealth();
    fetchFaultTypes();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/health`);
      setApiStatus({
        healthy: response.data.status === 'healthy',
        modelLoaded: response.data.model_loaded
      });
    } catch (err) {
      setApiStatus({ healthy: false, modelLoaded: false });
    }
  };

  const fetchFaultTypes = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/fault-types`);
      setFaultTypes(response.data.fault_types);
    } catch (err) {
      // Use default fault types if API is not available
      setFaultTypes([
        { name: 'Pastry', description: 'A type of surface defect characterized by flaky or layered appearance on the steel surface.' },
        { name: 'Z_Scratch', description: 'Linear scratches in a Z-shaped or zigzag pattern on the steel plate surface.' },
        { name: 'K_Scratch', description: 'K-shaped scratch marks typically caused during manufacturing or handling.' },
        { name: 'Stains', description: 'Discoloration or spots on the steel surface due to chemical reactions or contamination.' },
        { name: 'Dirtiness', description: 'Surface contamination from dust, oil, or other foreign particles.' },
        { name: 'Bumps', description: 'Raised areas or protrusions on the steel plate surface.' },
        { name: 'Other_Faults', description: 'Miscellaneous defects that do not fall into the above categories.' }
      ]);
    }
  };

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.gif']
    },
    multiple: false
  });

  const removeImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setResults(null);
    setError(null);
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      // Try the main predict endpoint first, fall back to demo
      let response;
      try {
        response = await axios.post(`${API_BASE_URL}/api/predict`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
      } catch (err) {
        // Fall back to demo endpoint
        response = await axios.post(`${API_BASE_URL}/api/demo-predict`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
      }

      if (response.data.success) {
        setResults(response.data);
      } else {
        setError(response.data.error || 'Analysis failed');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to connect to the server. Please ensure the backend is running.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <motion.div 
          className="logo-container"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <FaIndustry className="logo-icon" />
          <h1>Steel Plate Fault Detector</h1>
        </motion.div>
        <motion.p 
          className="subtitle"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
        >
          AI-Powered Defect Analysis using Convolutional Neural Networks
        </motion.p>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Status Indicator */}
        <motion.div 
          className="status-indicator"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <span className={`status-dot ${apiStatus.healthy ? 'healthy' : 'error'}`}></span>
          <span>
            {apiStatus.healthy 
              ? (apiStatus.modelLoaded ? 'API Connected • Model Ready' : 'API Connected • Demo Mode')
              : 'API Offline • Demo Mode Available'}
          </span>
        </motion.div>

        <div className="content-grid">
          {/* Upload Section */}
          <motion.div 
            className="card upload-card"
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="card-header">
              <div className="card-icon">
                <FaCloudUploadAlt />
              </div>
              <h2 className="card-title">Upload Image</h2>
            </div>

            {!selectedImage ? (
              <div 
                {...getRootProps()} 
                className={`upload-zone ${isDragActive ? 'drag-active' : ''}`}
              >
                <input {...getInputProps()} />
                <FaCloudUploadAlt className="upload-icon" />
                <p className="upload-text">
                  {isDragActive 
                    ? 'Drop the image here...' 
                    : 'Drag & drop a steel plate image here'}
                </p>
                <p className="upload-subtext">or click to select from your computer</p>
                <p className="upload-subtext">Supports: JPG, PNG, BMP, GIF</p>
              </div>
            ) : (
              <motion.div 
                className="image-preview-container"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
              >
                <img 
                  src={imagePreview} 
                  alt="Selected steel plate" 
                  className="image-preview"
                />
                <button 
                  className="remove-image-btn"
                  onClick={removeImage}
                  title="Remove image"
                >
                  <FaTimes />
                </button>
              </motion.div>
            )}

            <button 
              className="btn btn-primary analyze-btn"
              onClick={analyzeImage}
              disabled={!selectedImage || isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <span className="loading-spinner"></span>
                  Analyzing...
                </>
              ) : (
                <>
                  <FaSearch />
                  Analyze Image
                </>
              )}
            </button>

            {error && (
              <motion.div 
                className="error-message"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <FaExclamationTriangle />
                {error}
              </motion.div>
            )}
          </motion.div>

          {/* Results Section */}
          <motion.div 
            className="card results-card"
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="card-header">
              <div className="card-icon">
                <FaCheckCircle />
              </div>
              <h2 className="card-title">Analysis Results</h2>
            </div>

            <AnimatePresence mode="wait">
              {results ? (
                <motion.div 
                  className="results-section"
                  key="results"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                >
                  {results.demo_mode && (
                    <div className="demo-banner">
                      <FaInfoCircle />
                      Demo Mode - Results are simulated. Train the model for real predictions.
                    </div>
                  )}

                  <div className="primary-result">
                    <p className="result-label">Detected Fault Type</p>
                    <h3 className="result-fault-type">{results.primary_fault}</h3>
                    <p className="result-confidence">
                      {results.primary_confidence.toFixed(2)}% Confidence
                    </p>
                    <p className="result-description">
                      {results.primary_description}
                    </p>
                  </div>

                  <h4 style={{ marginBottom: '1rem', color: 'var(--text-secondary)' }}>
                    All Predictions
                  </h4>
                  <div className="predictions-list">
                    {results.all_predictions.map((prediction, index) => {
                      const IconComponent = faultIcons[prediction.fault_type] || FaQuestion;
                      return (
                        <motion.div 
                          className="prediction-item"
                          key={prediction.fault_type}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.1 }}
                        >
                          <div className="prediction-rank">{index + 1}</div>
                          <div className="prediction-info">
                            <div className="prediction-name">
                              <IconComponent style={{ marginRight: '8px', opacity: 0.7 }} />
                              {prediction.fault_type.replace('_', ' ')}
                            </div>
                            <div className="prediction-bar-container">
                              <motion.div 
                                className="prediction-bar"
                                initial={{ width: 0 }}
                                animate={{ width: `${prediction.confidence}%` }}
                                transition={{ delay: index * 0.1 + 0.2, duration: 0.5 }}
                              />
                            </div>
                          </div>
                          <span className="prediction-percentage">
                            {prediction.confidence.toFixed(1)}%
                          </span>
                        </motion.div>
                      );
                    })}
                  </div>
                </motion.div>
              ) : (
                <motion.div 
                  className="no-results"
                  key="no-results"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  style={{ 
                    textAlign: 'center', 
                    padding: '3rem 2rem',
                    color: 'var(--text-muted)'
                  }}
                >
                  <FaSearch style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.5 }} />
                  <p>Upload an image and click "Analyze" to detect faults</p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>

        {/* Fault Types Information */}
        <motion.section 
          className="fault-types-section"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
        >
          <div className="card">
            <div className="card-header">
              <div className="card-icon">
                <FaInfoCircle />
              </div>
              <h2 className="card-title">Types of Steel Plate Faults</h2>
            </div>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
              Our CNN model is trained to identify the following 7 types of defects in steel plates:
            </p>
            <div className="fault-types-grid">
              {faultTypes.map((fault, index) => {
                const IconComponent = faultIcons[fault.name] || FaQuestion;
                return (
                  <motion.div 
                    className="fault-type-card"
                    key={fault.name}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ scale: 1.02 }}
                  >
                    <h4 className="fault-type-name">
                      <IconComponent />
                      {fault.name.replace('_', ' ')}
                    </h4>
                    <p className="fault-type-description">{fault.description}</p>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </motion.section>
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>
          🏭 Steel Plate Fault Detector | Material Science Project | 
          Built with React, Flask & TensorFlow
        </p>
        <p style={{ marginTop: '0.5rem' }}>
          © 2026 | Powered by Convolutional Neural Networks
        </p>
      </footer>
    </div>
  );
}

export default App;

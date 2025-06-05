# Naruto Image Generator Frontend

This directory contains two frontend options for the Naruto Image Generator:

## 🌐 Option 1: HTML Frontend (Recommended for Netlify)

A modern, responsive HTML frontend that connects directly to your deployed backend.

### Files:
- `index.html` - Main application interface
- `style.css` - Beautiful, modern styling
- `netlify.toml` - Netlify deployment configuration

### Features:
- ✅ Responsive design that works on all devices
- ✅ Real-time parameter adjustment with sliders
- ✅ Image download functionality
- ✅ Error handling and loading states
- ✅ Direct API integration with your backend

### Deploy to Netlify:

1. **Connect Repository**: Link your GitHub repo to Netlify
2. **Configure Build**: 
   - Build command: `echo 'Static HTML deployment'`
   - Publish directory: `frontend`
   - Base directory: `frontend`
3. **Deploy**: Netlify will automatically deploy your site
4. **Environment**: The API URL is automatically configured

## 🖥️ Option 2: Streamlit Frontend

A full-featured Streamlit application with advanced UI components.

### Files:
- `app.py` - Streamlit application
- `config.py` - Configuration settings
- `utils.py` - Utility functions
- `requirements.txt` - Python dependencies

### Features:
- ✅ Rich interactive components
- ✅ Model comparison (finetuned vs base)
- ✅ Image gallery and metadata tracking
- ✅ Advanced parameter controls

### Deploy to Streamlit Cloud:

1. **Visit**: [streamlit.io](https://streamlit.io/)
2. **Connect**: Link your GitHub repository
3. **Configure**:
   - Main file: `frontend/app.py`
   - Python version: 3.10
   - Environment variable: `BACKEND_API_URL=https://naruto-image-generator.onrender.com`
4. **Deploy**: Streamlit Cloud will handle the rest

## 🚀 Quick Start (HTML Version)

1. Clone the repository
2. Navigate to the `frontend` directory
3. Open `index.html` in your browser
4. Start generating Naruto-style images!

## 🔧 Configuration

The backend API URL is configured in:
- **HTML**: Directly in the JavaScript (line 97 in `index.html`)
- **Streamlit**: In `config.py` via environment variable

## 📱 Mobile Support

Both frontends are fully responsive and work great on:
- 📱 Mobile phones
- 📱 Tablets  
- 💻 Desktop computers

## 🎨 Customization

The HTML frontend can be easily customized by:
- Modifying `style.css` for visual changes
- Updating `index.html` for functionality changes
- Adjusting the API URL for different backends

## 🔗 Backend Integration

Both frontends connect to: `https://naruto-image-generator.onrender.com`

The API endpoints used:
- `POST /generate` - Generate images
- `GET /health` - Health check
- `GET /docs` - API documentation

## 📞 Support

If you encounter any issues:
1. Check the browser console for errors
2. Verify the backend API is running
3. Ensure CORS is properly configured
4. Check network connectivity

---

**Recommendation**: Use the HTML frontend for Netlify deployment as it's lighter, faster, and doesn't require server-side Python execution. 
# 🚀 Streamlit Cloud Deployment Guide

## 📋 Prerequisites

1. **GitHub Account** - Your code needs to be in a GitHub repository
2. **Streamlit Cloud Account** - Sign up at [share.streamlit.io](https://share.streamlit.io)

## 🔑 Setting Up API Keys

### Option 1: Streamlit Cloud Secrets (Recommended)

1. **Create `.streamlit/secrets.toml`** in your repository:
```toml
OPENAI_API_KEY = "sk-your-actual-openai-key"
ANTHROPIC_API_KEY = "anthropic-your-actual-key"
GEMINI_API_KEY = "AIza-your-actual-gemini-key"
```

2. **Add to Streamlit Cloud:**
   - Go to your app settings in Streamlit Cloud
   - Navigate to "Secrets" section
   - Paste the contents of your `secrets.toml` file

### Option 2: Environment Variables

Set these in your Streamlit Cloud app settings:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY` 
- `GEMINI_API_KEY`

## 🚀 Deploy to Streamlit Cloud

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Add Streamlit app for deployment"
git push origin main
```

### Step 2: Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Set the path to your app: `streamlit_app.py`
5. Click "Deploy!"

### Step 3: Configure Secrets

1. In your app settings, go to "Secrets"
2. Add your API keys (see above)
3. Redeploy if needed

## 🔧 Configuration Options

### App Settings
- **Main file path:** `streamlit_app.py`
- **Python version:** 3.9+
- **Requirements file:** `requirements.txt`

### Environment Variables
- `STREAMLIT_SERVER_PORT`: 8501 (default)
- `STREAMLIT_SERVER_ADDRESS`: 0.0.0.0 (default)

## 📱 Features

### ✅ What Works in Cloud
- **Real LLM Analysis** - When API keys are provided
- **Simulation Mode** - Fallback when no keys available
- **Interactive Charts** - Plotly visualizations
- **Data Export** - CSV and text summaries
- **Responsive Design** - Works on all devices

### 🔄 Fallback Behavior
- **No API Keys** → Simulation mode with realistic scores
- **API Errors** → Automatic fallback to simulation
- **Missing Dependencies** → Graceful degradation

## 🧪 Testing

### Local Testing
```bash
streamlit run streamlit_app.py
```

### Cloud Testing
1. Deploy with simulation mode (no keys)
2. Test basic functionality
3. Add API keys and test real analysis
4. Verify all features work

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found"**
   - Check `requirements.txt` includes all packages
   - Ensure package names are correct

2. **"API key not found"**
   - Verify secrets are set in Streamlit Cloud
   - Check key names match exactly

3. **"App won't load"**
   - Check main file path is correct
   - Verify Python version compatibility

### Debug Mode
Add this to your app for debugging:
```python
st.write("Debug info:", st.secrets)
```

## 🔒 Security Notes

- **Never commit API keys** to your repository
- **Use Streamlit Cloud secrets** for sensitive data
- **API keys are encrypted** in Streamlit Cloud
- **Access logs** are available in your dashboard

## 📈 Scaling

- **Free tier:** 1 app, basic resources
- **Pro tier:** Multiple apps, more resources
- **Enterprise:** Custom deployment options

## 🎯 Next Steps

1. **Deploy your app**
2. **Test all features**
3. **Share with your team**
4. **Monitor usage and performance**
5. **Iterate and improve!**

---

**Need help?** Check the [Streamlit documentation](https://docs.streamlit.io) or [community forum](https://discuss.streamlit.io). 
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .crop-name {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('my_crop_model.pkl')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please ensure 'my_crop_model.pkl' is in the same directory.")
        st.stop()

model = load_model()

# Define crop labels (same order as in your training)
CROP_LABELS = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 
    'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 
    'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 
    'apple', 'orange', 'papaya', 'coconut', 'cotton', 
    'jute', 'coffee'
]

# Crop descriptions and information
CROP_INFO = {
    'rice': {
        'description': 'Rice is a staple food crop that thrives in flooded fields. It requires high water availability and warm temperatures.',
        'season': 'Kharif (June-October)',
        'duration': '3-6 months',
        'ideal_temp': '20-35°C',
        'water': 'High water requirement',
        'emoji': '🍚'
    },
    'maize': {
        'description': 'Maize (corn) is a versatile cereal crop used for food, feed, and industrial purposes. It adapts well to various climates.',
        'season': 'Kharif & Rabi',
        'duration': '3-4 months',
        'ideal_temp': '21-27°C',
        'water': 'Moderate water requirement',
        'emoji': '🌽'
    },
    'chickpea': {
        'description': 'Chickpea is a protein-rich legume crop that enriches soil with nitrogen. It grows well in cool, dry conditions.',
        'season': 'Rabi (October-March)',
        'duration': '4-5 months',
        'ideal_temp': '20-25°C',
        'water': 'Low to moderate water requirement',
        'emoji': '🫘'
    },
    'kidneybeans': {
        'description': 'Kidney beans are nutritious legumes rich in protein and fiber. They prefer moderate temperatures and well-drained soil.',
        'season': 'Kharif & Rabi',
        'duration': '3-4 months',
        'ideal_temp': '15-25°C',
        'water': 'Moderate water requirement',
        'emoji': '🫘'
    },
    'pigeonpeas': {
        'description': 'Pigeon peas are drought-resistant legumes that improve soil fertility. They are excellent for intercropping.',
        'season': 'Kharif (June-March)',
        'duration': '5-7 months',
        'ideal_temp': '20-30°C',
        'water': 'Low water requirement',
        'emoji': '🫛'
    },
    'mothbeans': {
        'description': 'Moth beans are drought-tolerant legumes ideal for arid regions. They require minimal water and enrich soil nitrogen.',
        'season': 'Kharif',
        'duration': '2-3 months',
        'ideal_temp': '25-35°C',
        'water': 'Very low water requirement',
        'emoji': '🫘'
    },
    'mungbean': {
        'description': 'Mung beans are fast-growing legumes rich in protein. They mature quickly and are excellent for crop rotation.',
        'season': 'Kharif & Summer',
        'duration': '2-3 months',
        'ideal_temp': '25-35°C',
        'water': 'Moderate water requirement',
        'emoji': '🫛'
    },
    'blackgram': {
        'description': 'Black gram (urad) is a protein-rich pulse crop. It grows well in warm conditions and enriches soil with nitrogen.',
        'season': 'Kharif & Rabi',
        'duration': '3-4 months',
        'ideal_temp': '25-35°C',
        'water': 'Moderate water requirement',
        'emoji': '🫘'
    },
    'lentil': {
        'description': 'Lentils are cool-season legumes high in protein. They fix nitrogen in soil and require minimal irrigation.',
        'season': 'Rabi (October-March)',
        'duration': '3-4 months',
        'ideal_temp': '18-25°C',
        'water': 'Low water requirement',
        'emoji': '🫘'
    },
    'pomegranate': {
        'description': 'Pomegranate is a drought-tolerant fruit crop rich in antioxidants. It thrives in semi-arid climates.',
        'season': 'Perennial',
        'duration': '2-3 years to fruiting',
        'ideal_temp': '20-30°C',
        'water': 'Low to moderate water requirement',
        'emoji': '🍎'
    },
    'banana': {
        'description': 'Banana is a tropical fruit crop requiring high moisture and warm temperatures. It provides year-round income.',
        'season': 'Year-round',
        'duration': '10-15 months',
        'ideal_temp': '25-35°C',
        'water': 'High water requirement',
        'emoji': '🍌'
    },
    'mango': {
        'description': 'Mango is the king of fruits, thriving in tropical climates. It requires minimal care once established.',
        'season': 'Summer fruiting',
        'duration': '3-5 years to fruiting',
        'ideal_temp': '24-30°C',
        'water': 'Moderate water requirement',
        'emoji': '🥭'
    },
    'grapes': {
        'description': 'Grapes are versatile fruit crops used for fresh consumption, wine, and raisins. They prefer warm, dry climates.',
        'season': 'Perennial',
        'duration': '2-3 years to fruiting',
        'ideal_temp': '15-25°C',
        'water': 'Moderate water requirement',
        'emoji': '🍇'
    },
    'watermelon': {
        'description': 'Watermelon is a refreshing summer fruit crop. It requires warm weather and well-drained sandy soil.',
        'season': 'Summer',
        'duration': '3-4 months',
        'ideal_temp': '25-35°C',
        'water': 'High water requirement',
        'emoji': '🍉'
    },
    'muskmelon': {
        'description': 'Muskmelon (cantaloupe) is a sweet summer fruit. It grows best in warm, sunny conditions with good drainage.',
        'season': 'Summer',
        'duration': '3-4 months',
        'ideal_temp': '25-35°C',
        'water': 'Moderate to high water requirement',
        'emoji': '🍈'
    },
    'apple': {
        'description': 'Apple is a temperate fruit crop requiring cold winters for proper fruiting. It thrives in hilly regions.',
        'season': 'Perennial',
        'duration': '3-5 years to fruiting',
        'ideal_temp': '15-25°C',
        'water': 'Moderate water requirement',
        'emoji': '🍎'
    },
    'orange': {
        'description': 'Orange is a citrus fruit crop rich in Vitamin C. It prefers subtropical climates with moderate rainfall.',
        'season': 'Perennial',
        'duration': '3-4 years to fruiting',
        'ideal_temp': '20-30°C',
        'water': 'Moderate water requirement',
        'emoji': '🍊'
    },
    'papaya': {
        'description': 'Papaya is a fast-growing tropical fruit crop. It starts fruiting within a year and requires warm conditions.',
        'season': 'Year-round',
        'duration': '9-12 months',
        'ideal_temp': '25-35°C',
        'water': 'High water requirement',
        'emoji': '🍈'
    },
    'coconut': {
        'description': 'Coconut is a versatile tropical crop providing food, oil, and fiber. It thrives in coastal areas with high rainfall.',
        'season': 'Perennial',
        'duration': '5-7 years to fruiting',
        'ideal_temp': '25-35°C',
        'water': 'High water requirement',
        'emoji': '🥥'
    },
    'cotton': {
        'description': 'Cotton is a major fiber crop used in textile industry. It requires warm weather and moderate rainfall.',
        'season': 'Kharif',
        'duration': '5-6 months',
        'ideal_temp': '21-30°C',
        'water': 'Moderate water requirement',
        'emoji': '🌱'
    },
    'jute': {
        'description': 'Jute is a natural fiber crop used for making sacks, rope, and textiles. It thrives in humid, warm conditions.',
        'season': 'Kharif',
        'duration': '4-5 months',
        'ideal_temp': '25-35°C',
        'water': 'High water requirement',
        'emoji': '🌾'
    },
    'coffee': {
        'description': 'Coffee is a shade-loving plantation crop. It requires cool, moist conditions and grows well in hilly regions.',
        'season': 'Perennial',
        'duration': '3-4 years to fruiting',
        'ideal_temp': '15-25°C',
        'water': 'Moderate to high water requirement',
        'emoji': '☕'
    }
}

# Feature columns (same order as training)
feature_cols = [
    'Nitrogen', 'Phosphorous', 'Potassium', 'temperature', 'humidity', 
    'ph', 'rainfall', 'N_P_ratio', 'N_K_ratio', 'P_K_ratio', 
    'Total_NPK', 'N_dominance', 'temp_category', 'humidity_category', 
    'rainfall_category', 'ph_category', 'temp_humidity', 'temp_rainfall'
]

# Prediction function
def predict_crop(nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall):
    # Create input dataframe
    input_data = pd.DataFrame([{
        'Nitrogen': nitrogen,
        'Phosphorous': phosphorous,
        'Potassium': potassium,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }])
    
    # Feature engineering
    input_data['N_P_ratio'] = input_data['Nitrogen'] / (input_data['Phosphorous'] + 1)
    input_data['N_K_ratio'] = input_data['Nitrogen'] / (input_data['Potassium'] + 1)
    input_data['P_K_ratio'] = input_data['Phosphorous'] / (input_data['Potassium'] + 1)
    input_data['Total_NPK'] = input_data['Nitrogen'] + input_data['Phosphorous'] + input_data['Potassium']
    input_data['N_dominance'] = input_data['Nitrogen'] / (input_data['Total_NPK'] + 1)
    
    # Temperature category
    if temperature <= 15:
        input_data['temp_category'] = 0
    elif temperature <= 25:
        input_data['temp_category'] = 1
    else:
        input_data['temp_category'] = 2
    
    # Humidity category
    if humidity <= 50:
        input_data['humidity_category'] = 0
    elif humidity <= 70:
        input_data['humidity_category'] = 1
    else:
        input_data['humidity_category'] = 2
    
    # Rainfall category
    if rainfall <= 100:
        input_data['rainfall_category'] = 0
    elif rainfall <= 200:
        input_data['rainfall_category'] = 1
    else:
        input_data['rainfall_category'] = 2
    
    # pH category
    if ph <= 5.5:
        input_data['ph_category'] = 0
    elif ph <= 7.5:
        input_data['ph_category'] = 1
    else:
        input_data['ph_category'] = 2
    
    # Interaction features
    input_data['temp_humidity'] = input_data['temperature'] * input_data['humidity']
    input_data['temp_rainfall'] = input_data['temperature'] * input_data['rainfall']
    
    # Ensure correct order
    input_data = input_data[feature_cols]
    
    # Predict
    prediction = model.predict(input_data)[0]
    crop_name = CROP_LABELS[prediction]
    
    result = {'recommended_crop': crop_name}
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_data)[0]
        confidence = probabilities[prediction] * 100
        
        # Top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3_crops = [CROP_LABELS[idx] for idx in top_3_idx]
        top_3_probs = probabilities[top_3_idx] * 100
        
        result['confidence'] = confidence
        result['top_3_predictions'] = list(zip(top_3_crops, top_3_probs))
    
    return result

# Header
st.markdown('<p class="main-header">🌾 Crop Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Get AI-powered crop recommendations based on soil and climate conditions</p>', unsafe_allow_html=True)

# Input form
st.markdown("### 📝 Enter Field Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Soil Nutrients (kg/ha)**")
    nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=90.0, step=1.0)
    phosphorous = st.number_input("Phosphorous (P)", min_value=5.0, max_value=145.0, value=42.0, step=1.0)
    potassium = st.number_input("Potassium (K)", min_value=5.0, max_value=205.0, value=43.0, step=1.0)

with col2:
    st.markdown("**Climate Conditions**")
    temperature = st.number_input("Temperature (°C)", min_value=8.0, max_value=43.0, value=20.8, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=14.0, max_value=100.0, value=82.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=300.0, value=202.9, step=0.1)

with col3:
    st.markdown("**Soil Properties**")
    ph = st.number_input("pH Level", min_value=3.5, max_value=10.0, value=6.5, step=0.1)
    st.markdown("")
    st.markdown("")
    predict_button = st.button("🌱 Get Recommendation", type="primary", use_container_width=True)

# Prediction
if predict_button:
    with st.spinner("🔄 Analyzing soil and climate data..."):
        result = predict_crop(nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall)
    
    # Display result
    st.markdown("---")
    st.markdown("### 🎯 Recommendation Result")
    
    crop_info = CROP_INFO.get(result['recommended_crop'], {})
    emoji = crop_info.get('emoji', '🌾')
    
    if 'confidence' in result:
        st.markdown(f"""
            <div class="result-box">
                <div style="font-size: 1.2rem; opacity: 0.9;">Recommended Crop</div>
                <div class="crop-name">{emoji} {result['recommended_crop'].upper()}</div>
                <div class="confidence">Confidence: {result['confidence']:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Crop information
        if crop_info:
            st.markdown("#### 📖 Crop Information")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:**")
                st.info(crop_info.get('description', 'No description available'))
            
            with col2:
                st.markdown(f"**Quick Facts:**")
                st.markdown(f"🗓️ **Season:** {crop_info.get('season', 'N/A')}")
                st.markdown(f"⏱️ **Duration:** {crop_info.get('duration', 'N/A')}")
                st.markdown(f"🌡️ **Temperature:** {crop_info.get('ideal_temp', 'N/A')}")
                st.markdown(f"💧 **Water:** {crop_info.get('water', 'N/A')}")
        
        # Top 3 predictions
        st.markdown("#### 📊 Alternative Recommendations")
        cols = st.columns(3)
        for i, (col, (crop, prob)) in enumerate(zip(cols, result['top_3_predictions'])):
            with col:
                crop_emoji = CROP_INFO.get(crop, {}).get('emoji', '🌾')
                st.metric(
                    label=f"#{i+1} Choice",
                    value=f"{crop_emoji} {crop.capitalize()}",
                    delta=f"{prob:.1f}%"
                )
    else:
        st.markdown(f"""
            <div class="result-box">
                <div style="font-size: 1.2rem; opacity: 0.9;">Recommended Crop</div>
                <div class="crop-name">{emoji} {result['recommended_crop'].upper()}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Crop information
        if crop_info:
            st.markdown("#### 📖 Crop Information")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:**")
                st.info(crop_info.get('description', 'No description available'))
            
            with col2:
                st.markdown(f"**Quick Facts:**")
                st.markdown(f"🗓️ **Season:** {crop_info.get('season', 'N/A')}")
                st.markdown(f"⏱️ **Duration:** {crop_info.get('duration', 'N/A')}")
                st.markdown(f"🌡️ **Temperature:** {crop_info.get('ideal_temp', 'N/A')}")
                st.markdown(f"💧 **Water:** {crop_info.get('water', 'N/A')}")
    
    # Input summary
    with st.expander("📋 View Input Summary"):
        summary_df = pd.DataFrame({
            'Parameter': ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
            'Value': [f"{nitrogen} kg/ha", f"{phosphorous} kg/ha", f"{potassium} kg/ha", 
                     f"{temperature}°C", f"{humidity}%", f"{ph}", f"{rainfall} mm"],
            'Status': ['✓'] * 7
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>💡 <b>Tip:</b> Ensure all measurements are accurate for best results</p>
        <p style="font-size: 0.9rem;">Powered by Machine Learning | Made with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)
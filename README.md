# Beyond Terra – Hunting Exoplanets with AI
(for link of this app refer to the bottom of this page)

Click here to checkout the demo video: https://drive.google.com/file/d/1Jde7AB81xBEck1nbezhHnB95Y2rN-4EX/view?usp=sharing

## Team Name & Members
**Team:** Beyond Terra  
**Members:** Khiyati Sharma, Kartik Gaur, Aprajita Sharma

---

## Challenge Name
**A World Away: Hunting for Exoplanets with AI**

---

## Short Description
Beyond Terra is an AI-powered tool designed to help researchers and enthusiasts identify exoplanets from NASA’s Kepler dataset. Using a Random Forest machine learning model trained on key stellar and orbital parameters, our app predicts whether a celestial object is likely a planet or not. The app also provides visual explanations, model accuracy metrics, and interactive insights to make exoplanet discovery accessible to beginners and experts alike.

---

## Slide Link
📄 **Slide Deck:** [Insert Google Drive / OneDrive Link]  

---

## Key Features
- **Exoplanet Prediction:** Enter stellar and orbital parameters to predict if an object is likely a planet.  
- **Model Accuracy Metrics:** View the model’s performance on the dataset, including classification report and confusion matrix.  
- **Feature Importance Visualization:** Identify which parameters most influence the model’s predictions.  
- **SHAP Analysis:** Explore individual feature contributions for each prediction.  
- **Interactive Interface:** Streamlit-based UI for a user-friendly experience.  

---

## How it Works
1. **Data Source:** Uses NASA’s Kepler exoplanet data (publicly available) for model training.  
2. **ML Model:** Random Forest classifier trained on labeled data (CONFIRMED vs FALSE POSITIVE/CANDIDATE).  
3. **Preprocessing:** Missing values handled via median imputation; data scaled using a standard scaler.  
4. **Prediction:** Users input observed features → model predicts probability of being a planet.  
5. **Visualizations:** Accuracy, confusion matrix, feature importance bar chart, and SHAP summary plot.  
6. **Interface:** Built with Streamlit for ease of use and interactive experience.  

---

## Future Work / Vision
- **Integrate Real-Time NASA Data:** Enable the app to fetch live exoplanet candidate data as it becomes available.  
- **Enhanced Model Performance:** Experiment with more complex ML models (XGBoost, Neural Networks) for higher accuracy.  
- **User Experience Improvements:** Add guided explanations for beginners and interactive tutorials.  
- **Open-Source Contribution:** Make the tool fully open-source so researchers and enthusiasts can contribute and improve it.  
- **Space Impact:** Expand the app to other datasets (e.g., TESS, JWST) to accelerate exoplanet discovery worldwide.  

# Run the Streamlit app
[streamlit run app.py
](https://beyondterra.streamlit.app/)

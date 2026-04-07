# AquaSense AI — Water Quality & Health Risk Predictor

AI-powered water quality monitoring for tier 2-3 cities in India.
Citizens upload a water photo + fill a form → AI predicts contamination type, risk level, and recommended action.

## Tech Stack
- Backend: Django + Django REST Framework + PostgreSQL + Celery
- AI: MobileNetV2 (CNN) + Random Forest + SHAP explainability
- Frontend: HTML/CSS/Bootstrap 5 + Leaflet.js heatmap

## Setup

### 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/aquasense-ai.git
cd aquasense-ai

### 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements/base.txt
pip install -r requirements/ml.txt

### 4. Run Day 1 ML pipeline
cd ml
python download_data.py
python preprocess.py
python train_rf.py
python train_cnn.py
python shap_explain.py
python predict.py

## Project Structure
See folder structure diagram in docs/
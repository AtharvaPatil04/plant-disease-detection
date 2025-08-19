# 🌱 Plant Disease Detection (LeafNet)

> A deep learning-powered system for detecting plant leaf diseases using Convolutional Neural Networks (CNNs) and deployed via Streamlit.

---

## 📌 Project Highlights

- ✅ Used a custom CNN model trained on the PlantVillage dataset  
- 📊 Achieved high validation accuracy across multiple plant classes  
- 📁 Provides detailed predictions:
  - Plant name  
  - Health status (Healthy/Diseased)  
  - Disease name  
  - Confidence score  
  - Scientific name  
  - Common name & cause (layman explanation)  
- 💻 Real-time predictions via Streamlit web app  
- 🚀 Ready for deployment and real-world usage  

---

## 🔧 Tech Stack

- Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn  
- Pillow (image handling)  
- Streamlit (frontend web app)  

---

## 📁 Folder Structure

```bash
plant-disease-detection/
├── data/                   ← (Not uploaded to GitHub)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation.ipynb
│   ├── 04_visualizations.ipynb
│   ├── 05_inference.py
│   └── 06_streamlit_app.py   ← Streamlit App
├── models/
│   └── leafnet_v1/
│       ├── best_model.h5
│       └── class_names.json
├── requirements.txt
├── README.md
├── .gitignore
```

---

## 🚀 How to Run the Streamlit App Locally

### 1. Clone the Repository
```bash
git clone https://github.com/AtharvaPatil04/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Create a Virtual Environment & Activate
```bash
python -m venv venv_tf
venv_tf\Scripts\activate   # For Windows
```

### 3. Dataset Used
The dataset used for this project is the **PlantVillage dataset** from Kaggle.  

📥 [Download Dataset from Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)

> It contains 54,000+ labeled leaf images of healthy and diseased plants.

### 4. Install Requirements
```bash
pip install -r requirements.txt
```

### 5. Download Model File
> 🔗 [Download best_model.h5 from Google Drive](https://drive.google.com/file/d/1hWOsxIUJMt_DcF_6oXRF0aWa014OKrwG/view?usp=sharing)  

- Place it in the `/models/leafnet_v1/` folder along with `class_names.json`

### 6. Run the Streamlit App
```bash
streamlit run notebooks/06_streamlit_app.py
```

---

## 📊 Sample Results

- Real-time predictions of plant leaf diseases  
- Outputs include plant, health status, disease, confidence %, scientific name, and cause  

---

## 📬 Contact

- 🔗 [LinkedIn](https://www.linkedin.com/in/atharvaajaypatil/)  
- 📧 atharvapatil221004@gmail.com  

---

## ⭐ Star this repo if you find it helpful!

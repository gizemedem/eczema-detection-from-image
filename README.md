# 🧠 Eczema Detection from Uploaded Images

This project uses a pre-trained deep learning model to detect signs of **eczema** from uploaded skin images. Built with **Python**, **TensorFlow**, and **OpenCV**, it allows users to upload a photo and get instant feedback on whether the skin appears to be normal or eczema-affected.

---

## 📸 Features

- Upload skin images in JPG, PNG format
- Predicts whether the image contains signs of eczema
- Built using a Convolutional Neural Network (CNN)
- Uses TensorFlow `.h5` model
- Simple and lightweight API using Flask

---

## 🚀 How It Works

1. User uploads a skin image via a web interface or API.
2. Image is resized and normalized.
3. The model makes a prediction based on trained data.
4. Output: `"Normal"` or `"Eczema"` (with probability score, optional).

---

## 🧪 Example Request

You can send a POST request to the endpoint:

```bash
POST /predict
Content-Type: multipart/form-data

📦 Installation
bash
Kopyala
Düzenle
pip install -r requirements.txt
python app.py


📁 Folder Structure
eczema-photo-detection/
├── app.py
├── model/
│   └── egzama_model.h5 (not uploaded to repo)
├── test_images/
│   └── test1.jpg
├── requirements.txt
├── README.md
├── .gitignore

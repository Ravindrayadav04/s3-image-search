# 🖼️ S3 Image Search (CLIP + LanceDB)

A powerful image search application built using **Streamlit**, **OpenAI CLIP**, **AWS S3**, and **LanceDB**.
It allows you to upload images, generate embeddings, and perform similarity-based search with color filtering.

---

# 🚀 Features

* 🔍 **Semantic Image Search** using CLIP embeddings
* 🎨 **Color-based Filtering** (dominant color matching)
* ☁️ **AWS S3 Integration** for image storage
* ⚡ **Fast Vector Search** using LanceDB
* 🖥️ **Interactive UI** with Streamlit
* 📦 **Batch Image Upload Support**
* 🧠 **Normalized Embeddings for Accurate Similarity**

---

# 📁 Project Structure

```
project/
│── app.py
│── .env
│── .gitignore
│── requirements.txt
│── assets/
│── lancedb/
```

---

# ⚙️ Setup & Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/s3-image-search.git
cd s3-image-search
```

---

## 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

### Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Setup Environment Variables

Create a `.env` file in the root directory:

```env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=eu-north-1
S3_BUCKET=your_bucket_name
```

---

# 🔐 Security Note

* Never commit `.env` to GitHub
* Ensure `.env` is listed in `.gitignore`
* Rotate AWS keys if exposed

---

# ☁️ AWS Setup

1. Create an **S3 Bucket**
2. Enable proper permissions (read/write)
3. Create IAM user with:

   * `AmazonS3FullAccess` (or scoped policy)

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 🧪 How It Works

## Upload Flow

1. Upload images via UI
2. Images stored in S3
3. CLIP generates embeddings
4. Colors extracted using KMeans
5. Data stored in LanceDB

---

## Search Flow

1. Upload query image
2. Generate embedding
3. Perform vector similarity search
4. Apply:

   * Color distance filter
   * Cosine similarity threshold
5. Display matching images

---

# 🧠 Tech Stack

* **Frontend:** Streamlit
* **ML Model:** CLIP (ViT-B/32)
* **Vector DB:** LanceDB
* **Storage:** AWS S3
* **Processing:** NumPy, Scikit-learn

---

# 📊 Key Concepts Used

* Cosine Similarity
* Image Embeddings
* KMeans Clustering (color extraction)
* Vector Search

---

# ⚡ Performance Tips

* Use smaller images for faster processing
* Adjust similarity threshold (`0.75`) for better results
* Tune color distance (`80`) for stricter filtering

---

# 🐛 Troubleshooting

### ❌ Images not loading

* Check AWS credentials
* Verify S3 bucket permissions

### ❌ No search results

* Lower similarity threshold
* Check embedding generation

### ❌ App not starting

* Ensure all dependencies installed
* Activate virtual environment
  
---

# 👨‍💻 Author

Built with ❤️ for scalable image search systems.

---

# ⭐ If you like this project

Give it a ⭐ on GitHub and share it!

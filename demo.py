import streamlit as st
import boto3
import torch
import clip
import numpy as np
from PIL import Image
import io
import lancedb
import pyarrow as pa
from sklearn.cluster import KMeans
import os
from dotenv import load_dotenv

# =========================
# LOAD ENV
# =========================
load_dotenv()

UPLOAD_BUCKET = os.getenv("S3_BUCKET")

# =========================
# AWS S3 CLIENT
# =========================
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

# =========================
# DEVICE (GPU/CPU)
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

model, preprocess = load_model()

# =========================
# LANCEDB SETUP
# =========================
db = lancedb.connect("./lancedb")
table_name = "image_vectors"

def init_db():
    try:
        # Try opening table
        return db.open_table(table_name)

    except Exception:
        # If not exists → create it
        schema = pa.schema([
            ("id", pa.string()),
            ("embedding", pa.list_(pa.float32(), 512)),
            ("path", pa.string()),
            ("metadata", pa.struct([
                ("primary_color", pa.string()),
                ("colors", pa.list_(pa.string()))
            ]))
        ])

        return db.create_table(table_name, schema=schema)

table = init_db()

# =========================
# COLOR EXTRACTION
# =========================
def extract_colors(image, k=3):
    try:
        image = image.resize((100, 100))
        arr = np.array(image).reshape((-1, 3))

        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(arr)

        colors = kmeans.cluster_centers_.astype(int)
        color_list = [f"{c[0]}-{c[1]}-{c[2]}" for c in colors]

        return color_list[0], color_list
    except:
        return "0-0-0", []

# =========================
# COLOR DISTANCE
# =========================
def color_distance(c1, c2):
    try:
        r1, g1, b1 = map(int, c1.split("-"))
        r2, g2, b2 = map(int, c2.split("-"))
        return np.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)
    except:
        return 999

# =========================
# EMBEDDING
# =========================
def get_embedding(image):
    image = image.convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(image_tensor)

    emb = emb[0].cpu().numpy()
    emb = emb / np.linalg.norm(emb)

    return emb.astype(np.float32)

# =========================
# S3 IMAGE FETCH
# =========================
@st.cache_data
def get_s3_image(key):
    try:
        obj = s3.get_object(Bucket=UPLOAD_BUCKET, Key=key)
        return Image.open(io.BytesIO(obj["Body"].read()))
    except:
        return None

# =========================
# THUMBNAIL
# =========================
def show_thumbnail(image):
    img = image.copy()
    img.thumbnail((150, 150))
    return img

# =========================
# UI
# =========================
st.title("🖼️ Image Search (S3 + LanceDB + CLIP)")

option = st.selectbox("Choose Action", ["Upload Images", "Search Image"])

# =========================
# UPLOAD
# =========================
if option == "Upload Images":

    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            try:
                file_name = file.name
                file_bytes = file.read()

                # Upload to S3
                s3.put_object(
                    Bucket=UPLOAD_BUCKET,
                    Key=file_name,
                    Body=file_bytes
                )

                image = Image.open(io.BytesIO(file_bytes))

                emb = get_embedding(image)
                primary_color, all_colors = extract_colors(image)

                table.add([{
                    "id": file_name,
                    "embedding": emb.tolist(),
                    "path": file_name,
                    "metadata": {
                        "primary_color": primary_color,
                        "colors": all_colors
                    }
                }])

                st.image(show_thumbnail(image), caption=file_name)
                st.success(f"Uploaded: {file_name}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# =========================
# SEARCH
# =========================
elif option == "Search Image":

    query_file = st.file_uploader("Upload query image")

    if query_file:
        query_bytes = query_file.read()
        query_image = Image.open(io.BytesIO(query_bytes))

        st.image(show_thumbnail(query_image), caption="Query")

        query_emb = get_embedding(query_image)
        query_primary, _ = extract_colors(query_image)

        results = table.search(query_emb).limit(30).to_list()

        st.subheader("Results")

        found = False

        for r in results:
            metadata = r.get("metadata", {})
            db_color = metadata.get("primary_color")

            # Color filter
            if db_color:
                dist = color_distance(query_primary, db_color)
                if dist > 100:
                    continue

            db_emb = np.array(r["embedding"], dtype=np.float32)
            similarity = np.dot(query_emb, db_emb)

            if similarity < 0.7:
                continue

            img = get_s3_image(r["path"])
            if img is None:
                continue

            found = True

            st.image(
                show_thumbnail(img),
                caption=f"{r['path']} | Score: {similarity:.2f}"
            )

        if not found:
            st.warning("No good match found")
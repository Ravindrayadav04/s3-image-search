print("File started")

import boto3
import os
import json
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

UPLOAD_BUCKET = os.getenv("S3_BUCKET")

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

# Load CLIP model
model, preprocess = clip.load("ViT-B/32")


def find_folder(folder_name):
    """Find folder in current directory ignoring case"""
    cwd = os.getcwd()
    for f in os.listdir(cwd):
        if os.path.isdir(f) and f.lower() == folder_name.lower():
            return f
    return None


def upload_folder(folder):
    """Upload all images in folder to S3, including subfolders"""
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(root, file)
                key = os.path.relpath(path, folder)  # preserve folder structure
                s3.upload_file(path, BUCKET, key)
                print("Uploaded:", key)


def get_embedding(path):
    """Generate CLIP embedding for a single image"""
    image = Image.open(path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_image(image_tensor)
    return emb[0].numpy()


def create_index(folder):
    """Create embeddings for all images and save each embedding individually to S3"""
    print("Scanning folder:", folder)
    total = 0

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(root, file)
                print("Processing:", path)
                emb = get_embedding(path).tolist()

                # Save each embedding as a separate JSON in S3
                emb_key = os.path.relpath(path, folder) + ".json"  # e.g., single/img1.jpg.json
                s3.put_object(
                    Bucket=BUCKET,
                    Key=emb_key,
                    Body=json.dumps({"file": path, "embedding": emb}, indent=4)
                )
                total += 1

    if total == 0:
        print("No images found! Check your folder path or file extensions.")
    else:
        print(f"\nTotal images processed and embeddings uploaded: {total}")


def search(query_path, top_k=5):
    """Search similar images using embeddings stored individually in S3 and show them"""
    query_emb = get_embedding(query_path)

    # List embedding JSON files in S3
    response = s3.list_objects_v2(Bucket=BUCKET)
    objects = response.get("Contents", [])
    embeddings_files = [obj['Key'] for obj in objects if obj['Key'].endswith(".json")]

    results = []

    for emb_key in embeddings_files:
        # Load embedding JSON from S3
        obj = s3.get_object(Bucket=BUCKET, Key=emb_key)
        content = obj['Body'].read()
        data = json.loads(content)

        # Check if the JSON is valid
        if isinstance(data, dict) and "embedding" in data:
            sim = cosine_similarity([query_emb], [np.array(data["embedding"])])[0][0]
            results.append((data["file"], sim, emb_key.replace(".json", "")))
        else:
            print(f"Skipping invalid embedding file: {emb_key}")

    if not results:
        print("No embeddings found in S3 for search!")
        return

    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)

    print("\nTop Matches:")
    for r in results[:top_k]:
        print(f"{r[0]} -> Similarity: {r[1]:.4f}")

    # Show top images visually
    fig, axes = plt.subplots(1, min(top_k, len(results)), figsize=(4*min(top_k, len(results)), 4))
    if top_k == 1 or len(results) == 1:
        axes = [axes]

    for i, (file_path, sim, img_key) in enumerate(results[:top_k]):
        obj = s3.get_object(Bucket=BUCKET, Key=img_key)  # get actual image
        img = Image.open(io.BytesIO(obj['Body'].read()))
        axes[i].imshow(img)
        axes[i].set_title(f"{sim:.2f}")
        axes[i].axis('off')

    plt.show()


if __name__ == "__main__":
    folder = find_folder("assets")
    if not folder:
        print("Error: 'assets' folder not found in current directory!")
        exit()

    print("1. Upload images to S3")
    print("2. Create embeddings index (S3 vector-style)")
    print("3. Search similar images")

    choice = input("Enter choice: ")

    if choice == "1":
        upload_folder(folder)

    elif choice == "2":
        create_index(folder)

    elif choice == "3":
        query = input("Enter query image path: ")
        search(query)
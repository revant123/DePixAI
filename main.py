import cv2
import numpy as np
import joblib
import os

def preprocess_image(image, downscale_factor=2):
    height, width = image.shape[:2]
    new_size = (width // downscale_factor, height // downscale_factor)
    downscaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return downscaled_image

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def analyze_blocks(edges, block_size=16):
    height, width = edges.shape
    block_scores = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = edges[y:y+block_size, x:x+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                variance = np.var(block)
                edge_density = np.sum(block) / (block_size * block_size)
                block_scores.append((variance, edge_density))
    return block_scores

model = joblib.load('model/pixelation_model.pkl')

def predict_pixelation(block_scores):
    features = np.array(block_scores)
    predictions = model.predict(features)
    return np.mean(predictions)

def is_image_pixelated(image_path, downscale_factor=2, block_size=16, threshold=0.5):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unable to open: {image_path}")
    
    downscaled_img = preprocess_image(img, downscale_factor)
    edges = detect_edges(downscaled_img)
    block_scores = analyze_blocks(edges, block_size)
    pixelation_score = predict_pixelation(block_scores)
    is_pixelated = pixelation_score > threshold
    return is_pixelated

image_path1 = 'input_images/pixelated-cloud.webp'
image_path2 = 'input_images/non.jpg'
result1 = is_image_pixelated(image_path1)
result2 = is_image_pixelated(image_path2)
print(f'Final result for {image_path1}: {"Pixelated" if result1 else "Not Pixelated"}')
print(f'Final result for {image_path2}: {"Pixelated" if result2 else "Not Pixelated"}')

import cv2
import numpy as np

def zoom_in_image(image, scale=4):
    height, width = image.shape[:2]
    new_size = (width * scale, height * scale)
    zoomed_image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
    return zoomed_image

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def is_image_pixelated(image_path, threshold=10, block_size=16):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unable to open: {image_path}")

    zoomed_img = zoom_in_image(img)
    edges = detect_edges(zoomed_img)
    blocky_score = 0
    height, width = edges.shape
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = edges[y:y+block_size, x:x+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                variance = np.var(block)
                edge_density = np.sum(block) / (block_size * block_size)
                blocky_score += variance * edge_density

    num_blocks = (height // block_size) * (width // block_size)
    avg_blocky_score = blocky_score / num_blocks
    print(f"Blocky score for {image_path}: {avg_blocky_score}")
    is_pixelated = avg_blocky_score < threshold
    print(f"Is {image_path} pixelated? {is_pixelated}")
    return is_pixelated

image_path1 = 'input_images/pixelated-cloud.webp'
image_path2 = 'input_images/non.jpg'

result1 = is_image_pixelated(image_path1)
result2 = is_image_pixelated(image_path2)

print(f"Final result for {image_path1}: {'Pixelated' if result1 else 'Not Pixelated'}")
print(f"Final result for {image_path2}: {'Pixelated' if result2 else 'Not Pixelated'}")

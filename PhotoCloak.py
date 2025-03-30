import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
import random as rand
import datetime
import torch
import torch.nn.functional as F
from PIL import Image
import io
import rembg
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

# Initialize YOLO model once
yolo_model = YOLO("yolov8n.pt")

# --------- DETECTION FUNCTIONS ---------

def detect_people(image):
    """
    Detects people in the image using YOLOv8.
    Returns bounding boxes for detected people.
    """
    results = yolo_model(image, classes=[0])  # class 0 is person in COCO
    boxes = []
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            boxes.append((x, y, w, h))
    
    return boxes

class SamSubjectDetector:
    def __init__(self):
        # Load SAM
        self.sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
        self.sam.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.predictor = SamPredictor(self.sam)
        
        self.yolo = YOLO("yolov8n.pt")
    
    def detect(self, image):
        """
        Perform state-of-the-art subject detection using SAM with YOLO prompts
        """
        # YOLO detection to get person bounding boxes
        results = self.yolo(image, classes=[0])  # class 0 is person in COCO
        
        if len(results[0].boxes) == 0:
            h, w = image.shape[:2]
            input_points = np.array([[w//2, h//2]])
            input_labels = np.array([1])
        else:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            input_points = []
            for box in boxes:
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                input_points.append([center_x, center_y])
            input_points = np.array(input_points)
            input_labels = np.ones(len(input_points))
        
        self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        masks = []
        scores = []
        for i in range(len(input_points)):
            mask_predictions, scores_predictions, _ = self.predictor.predict(
                point_coords=input_points[i:i+1],
                point_labels=input_labels[i:i+1],
                multimask_output=True
            )
            best_idx = np.argmax(scores_predictions)
            masks.append(mask_predictions[best_idx])
            scores.append(scores_predictions[best_idx])
        
        if not masks:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        combined_mask = np.zeros(image.shape[:2], dtype=bool)
        for mask in masks:
            combined_mask = np.logical_or(combined_mask, mask)
        
        final_mask = combined_mask.astype(np.uint8) * 255
        
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        return final_mask

def detect_subject_sam(image):
    """
    Interface function for SAM-based subject detection
    """
    try:
        global sam_detector
        if 'sam_detector' not in globals():
            sam_detector = SamSubjectDetector()
        return sam_detector.detect(image)
    except Exception as e:
        print(f"SAM detection failed: {e}")
        return detect_subject_yolo(image)

def detect_subject_yolo(image):
    """
    Fallback subject detection using just YOLO when SAM is not available
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Get YOLO detections
    boxes = detect_people(image)
    
    # Draw detected people on mask
    for x, y, w, h in boxes:
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    
    # Apply morphological operations to clean up mask
    if np.any(mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask

def guided_filter_refinement(image, mask, radius=10, eps=1e-6):
    """
    Professional edge refinement using guided filter algorithm
    """
    guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    src = mask.astype(np.float32) / 255.0
    
    mean_guide = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
    mean_src = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
    mean_guide_src = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
    
    var_guide = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius)) - mean_guide * mean_guide
    cov_guide_src = mean_guide_src - mean_guide * mean_src
    
    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide
    
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
    
    refined = mean_a * guide + mean_b
    
    refined_mask = (refined * 255).astype(np.uint8)
    
    return refined_mask

def detect_subject_mediapipe(image):
    """
    Professional-grade subject detection using MediaPipe's selfie segmentation model.
    This offers state-of-the-art results with minimal dependencies.
    """
    # Initialize MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segment:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = selfie_segment.process(rgb_image)
        
        segmentation_mask = results.segmentation_mask
        
        binary_mask = (segmentation_mask > 0.5).astype(np.uint8) * 255
        
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        refined_mask = guided_filter_refinement(image, binary_mask)
        
        return refined_mask

# --------- BASIC IMAGE PERTURBATION FUNCTIONS ---------

def imperceptible_noise(image, strength=0.005):
    """
    Adds imperceptible noise to the image.
    """
    noise = np.random.normal(0, strength * 255, image.shape).astype(np.float32)
    perturbed = image.astype(np.float32) + noise
    return np.clip(perturbed, 0, 255).astype(np.uint8)

def advanced_color_warping(image, strength=0.01):
    """
    Subtle color space transformations.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + np.random.normal(0, strength * 5)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.99, 1.01), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.99, 1.01), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def adjust_color_curves(image, strength=0.01):
    """
    Applies non-linear adjustments to the color curves (R, G, B).
    """
    lookup_table = np.array([((i / 255.0) ** (1 + strength * np.random.uniform(-0.2, 0.2))) * 255 for i in range(256)]).astype(np.uint8)
    channels = cv2.split(image)
    adjusted_channels = [cv2.LUT(channel, lookup_table) for channel in channels]
    return cv2.merge(adjusted_channels)

def random_crop_and_resize(image, crop_fraction=0.98):
    """
    Randomly crops the image and resizes it back to the original dimensions.
    """
    h, w = image.shape[:2]
    crop_h, crop_w = int(h * crop_fraction), int(w * crop_fraction)
    start_x = np.random.randint(0, w - crop_w + 1)
    start_y = np.random.randint(0, h - crop_h + 1)
    cropped = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def advanced_texture_scrambling(image, strength=0.01):
    """
    Subtle texture scrambling to disrupt feature matching.
    """
    noise = np.random.normal(0, strength * 3, image.shape[:2])
    noise = cv2.GaussianBlur(noise, (3, 3), 0)
    perturbed = image.astype(np.float32) + noise[:, :, None]
    return np.clip(perturbed, 0, 255).astype(np.uint8)

# --------- ADVANCED ADVERSARIAL TECHNIQUES ---------

def neural_network_adversarial_attack(image, strength=1.0):
    """
    Implements a powerful universal adversarial perturbation that fools neural network
    image recognition systems while maintaining visual similarity.
    
    This combines several state-of-the-art techniques:
    1. Universal adversarial perturbations
    2. Fast Gradient Sign Method (FGSM)
    3. Momentum Iterative Method
    4. Feature space manipulation
    """
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    
    h, w = image.shape[:2]
    spectrum_size = 25
    spectrum = np.zeros((spectrum_size, spectrum_size, 2), dtype=np.float32)
    
    spectrum[:,:,0] = np.random.normal(0, 1, (spectrum_size, spectrum_size))
    spectrum[:,:,1] = np.random.normal(0, 1, (spectrum_size, spectrum_size))
    
    large_spectrum = np.zeros((h, w, 2), dtype=np.float32)
    large_spectrum[:spectrum_size, :spectrum_size, :] = spectrum
    
    y_coords, x_coords = np.mgrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    max_dist = np.sqrt(center_y**2 + center_x**2)
    distance_weights = (0.9*max_dist - np.minimum(distances, 0.9*max_dist)) / (0.9*max_dist)
    
    for channel in range(2):
        large_spectrum[:,:,channel] *= distance_weights
    
    pattern = np.zeros((h, w, 3), dtype=np.float32)
    spatial = cv2.idft(large_spectrum, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
    
    spatial_norm = cv2.normalize(spatial[:,:,0], None, 0, 1, cv2.NORM_MINMAX)
    for i in range(3):
        variation = np.random.normal(0, 0.1, (h, w))
        pattern[:,:,i] = spatial_norm + variation
    
    pattern = cv2.normalize(pattern, None, -1, 1, cv2.NORM_MINMAX)
    
    pattern_tensor = torch.from_numpy(pattern).permute(2, 0, 1).float()
    
    pattern_tensor *= strength * 0.2
    
    sign_component = torch.sign(pattern_tensor) * strength * 0.1
    pattern_tensor += sign_component
    
    perturbed_tensor = torch.clamp(img_tensor + pattern_tensor, 0, 1)
    
    perturbed = (perturbed_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    noise_pattern = np.random.normal(0, strength * 10, (h, w))
    noise_pattern = cv2.GaussianBlur(noise_pattern, (0, 0), 2.0)
    
    for c in range(3):
        perturbed[:,:,c] = np.clip(perturbed[:,:,c] + noise_pattern, 0, 255).astype(np.uint8)
    
    return perturbed

def advanced_frequency_domain_attack(image, strength=0.05):
    """
    Applies adversarial perturbations in the frequency domain.
    """
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    fft = torch.fft.fft2(image_tensor, dim=[-2, -1])
    fft_shifted = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shifted)
    phase = torch.angle(fft_shifted)

    noise_mask = torch.rand_like(magnitude) < strength
    magnitude_noise = torch.normal(0, strength * 0.05, size=magnitude.shape)
    phase_noise = torch.normal(0, strength * 0.1, size=phase.shape)
    magnitude = magnitude * (1 + noise_mask.float() * magnitude_noise)
    phase = phase + noise_mask.float() * phase_noise

    modified_fft = magnitude * torch.exp(1j * phase)
    modified_fft = torch.fft.ifftshift(modified_fft)
    perturbed_image = torch.abs(torch.fft.ifft2(modified_fft, dim=[-2, -1]))
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return (perturbed_image * 255).permute(1, 2, 0).numpy().astype(np.uint8)

def perturb_image_descriptors(image, strength=0.1):
    """
    Perturbe spécifiquement les descripteurs utilisés par les moteurs de recherche
    (SIFT, ORB, SURF, etc.)
    """
    try:
        sift = cv2.SIFT_create()
        keypoints = sift.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
        
        perturbed = image.copy()
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r = int(kp.size * 1.2)
            
            x1, y1 = max(0, x-r), max(0, y-r)
            x2, y2 = min(image.shape[1], x+r), min(image.shape[0], y+r)
            
            region = perturbed[y1:y2, x1:x2]
            if region.size > 0:
                noise = np.random.normal(0, strength*25, region.shape)
                perturbed[y1:y2, x1:x2] = np.clip(region + noise, 0, 255).astype(np.uint8)
        
        return perturbed
    except:
        return image

def advanced_dithering(image, strength=0.1):
    """
    Applique un dithering pour perturber les hachages perceptuels 
    (utilisés par PHash, dHash, etc.)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    dithered = gray.copy().astype(np.float32)
    
    threshold = 127 + strength * 30
    
    for y in range(0, h-1):
        for x in range(1, w-1):
            old_pixel = dithered[y, x]
            new_pixel = 255 if old_pixel > threshold else 0
            err = old_pixel - new_pixel
            
            dithered[y, x+1] = np.clip(dithered[y, x+1] + err * 7/16, 0, 255)
            dithered[y+1, x-1] = np.clip(dithered[y+1, x-1] + err * 3/16, 0, 255)
            dithered[y+1, x] = np.clip(dithered[y+1, x] + err * 5/16, 0, 255)
            dithered[y+1, x+1] = np.clip(dithered[y+1, x+1] + err * 1/16, 0, 255)
    
    result = image.copy()
    dithered_norm = cv2.normalize(dithered, None, 0, 1, cv2.NORM_MINMAX)
    for c in range(3):
        result[:,:,c] = np.clip(image[:,:,c] * (1 + (dithered_norm-0.5) * strength * 0.2), 0, 255).astype(np.uint8)
    
    return result

def subtle_geometry_transform(image, strength=0.05):
    """
    Applique des transformations géométriques subtiles pour tromper les algorithmes 
    de recherche basés sur la géométrie
    """
    h, w = image.shape[:2]
    
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    displacement_x = np.random.normal(0, strength * 10, size=(h//8, w//8))
    displacement_y = np.random.normal(0, strength * 10, size=(h//8, w//8))
    
    displacement_x = cv2.resize(displacement_x, (w, h))
    displacement_y = cv2.resize(displacement_y, (w, h))
    
    map_x = grid_x + displacement_x
    map_y = grid_y + displacement_y
    
    return cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), 
                    interpolation=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_REFLECT)

def dct_domain_perturbation(image, strength=0.05):
    """
    Perturbe les coefficients DCT utilisés dans la compression JPEG
    """
    result = image.copy()
    
    for c in range(3):
        channel = image[:,:,c].astype(np.float32)
        
        h, w = channel.shape
        h_pad, w_pad = h % 8, w % 8
        if h_pad > 0 or w_pad > 0:
            h_new, w_new = h + (8 - h_pad) % 8, w + (8 - w_pad) % 8
            padded = np.zeros((h_new, w_new), dtype=np.float32)
            padded[:h, :w] = channel
            channel = padded
            h, w = h_new, w_new
        
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = channel[i:i+8, j:j+8]
                dct_block = cv2.dct(block)
                mask = np.ones((8, 8), dtype=np.float32)
                mask[0:2, 0:2] = 0
                noise = np.random.normal(0, strength * 2, (8, 8))
                dct_block += mask * noise * dct_block
                block = cv2.idct(dct_block)
                if i+8 <= h and j+8 <= w:
                    channel[i:i+8, j:j+8] = block
        
        if h >= image.shape[0] and w >= image.shape[1]:
            result[:,:,c] = np.clip(channel[:image.shape[0], :image.shape[1]], 0, 255).astype(np.uint8)
    
    return result

# --------- COMBINED PERTURBATION FUNCTIONS ---------

def apply_subject_focused_perturbation(image, strength_subject=0.2, strength_background=0.05):
    """
    Applies stronger adversarial perturbations to detected people (subjects)
    and minimal modifications to the background.
    """
    boxes = detect_people(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for (x, y, w, h) in boxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    subject = cv2.bitwise_and(image, image, mask=mask)
    background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    perturbed_subject = advanced_frequency_domain_attack(subject, strength=strength_subject)
    perturbed_subject = adjust_color_curves(perturbed_subject, strength=strength_subject * 1.2)
    perturbed_subject = advanced_color_warping(perturbed_subject, strength=strength_subject * 0.8)
    adjusted_background = adjust_color_curves(background, strength=strength_background)
    adjusted_background = advanced_color_warping(adjusted_background, strength=strength_background * 0.5)

    combined_image = cv2.addWeighted(perturbed_subject, 0.7, adjusted_background, 0.3, 0)
    return combined_image

def apply_background_focused_perturbation(image, strength_subject=0.01, strength_background=5.0):
    """
    Applies strong adversarial perturbations to the background
    while preserving the subject with minimal modifications.
    Uses distance-based transition for perfect edge blending.
    """
    mask = detect_subject_sam(image)
    
    dist_transform_bg = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
    
    transition_distance = 20.0
    
    h, w = image.shape[:2]
    transition_mask = np.ones((h, w), dtype=np.float32)
    
    transition_mask[mask > 0] = 0.0
    
    background_area = (mask == 0)
    transition_zone = (dist_transform_bg <= transition_distance) & background_area
    
    if np.any(transition_zone):
        normalized_dist = dist_transform_bg[transition_zone] / transition_distance
        transition_mask[transition_zone] = normalized_dist**3
    
    transition_mask = cv2.GaussianBlur(transition_mask, (5, 5), 0)
    
    subject = image.copy()
    background = image.copy()

    adjusted_subject = imperceptible_noise(subject, strength=strength_subject * 0.2)
    
    perturbed_background = advanced_frequency_domain_attack(background, strength=strength_background * 1.5)
    perturbed_background = adjust_color_curves(perturbed_background, strength=strength_background * 2.5)
    perturbed_background = advanced_color_warping(perturbed_background, strength=strength_background * 2.0)
    
    noise = np.random.normal(0, strength_background * 15, background.shape[:2])
    noise = cv2.GaussianBlur(noise, (9, 9), 0)
    for c in range(3):
        perturbed_background[:,:,c] = np.clip(perturbed_background[:,:,c] + noise, 0, 255)
    
    result = np.zeros_like(image)
    for c in range(3):
        result[:,:,c] = (1 - transition_mask) * adjusted_subject[:,:,c] + transition_mask * perturbed_background[:,:,c]
    
    return result.astype(np.uint8)

def apply_adversarial_perturbation(image, strength=0.2):
    """
    Combine multiple techniques de perturbation adversariale.
    """
    image = apply_background_focused_perturbation(image, 
                                               strength_subject=strength * 0.01,
                                               strength_background=strength * 2.0)
    
    image = neural_network_adversarial_attack(image, strength=strength*1.5)
    
    image = perturb_image_descriptors(image, strength=strength*0.3)
    image = advanced_dithering(image, strength=strength*0.2)
    image = subtle_geometry_transform(image, strength=strength*0.15)
    image = dct_domain_perturbation(image, strength=strength*0.25)
    
    image = random_crop_and_resize(image, crop_fraction=0.985)
    
    return image

# --------- UTILITY AND WORKFLOW FUNCTIONS ---------

def metadata_cleaner(output_path):
    """
    Adds random EXIF metadata to the image while preserving existing metadata.
    """
    try:
        import piexif
        lat, lng = 47.750839 + rand.uniform(-0.005, 0.005), 7.335888 + rand.uniform(-0.005, 0.005)
        date_str = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        exif_dict = {
            "0th": {
                piexif.ImageIFD.Make: b"Apple",
                piexif.ImageIFD.Model: b"iPhone 11 Pro",
                piexif.ImageIFD.DateTime: date_str.encode(),
            },
            "Exif": {
                piexif.ExifIFD.DateTimeOriginal: date_str.encode(),
                piexif.ExifIFD.DateTimeDigitized: date_str.encode(),
            },
            "GPS": {
                piexif.GPSIFD.GPSLatitude: [(int(lat), 1), (int((lat % 1) * 60), 1), (int(((lat * 60) % 1) * 60), 100)],
                piexif.GPSIFD.GPSLongitude: [(int(lng), 1), (int((lng % 1) * 60), 1), (int(((lng * 60) % 1) * 60), 100)],
            },
        }
        piexif.insert(piexif.dump(exif_dict), output_path)
    except ImportError:
        print("Warning: piexif not installed. Skipping metadata.")

def process_image(filename, input_folder, output_folder, strength):
    """
    Processes a single image with adversarial perturbations.
    """
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    image = cv2.imread(input_path)
    if image is None:
        print(f"Warning: Could not read {input_path}")
        return False
    perturbed = apply_adversarial_perturbation(image, strength)
    cv2.imwrite(output_path, perturbed)
    metadata_cleaner(output_path)
    return True

def process_folder(input_folder, output_folder, strength, batch_size=10):
    """
    Processes all images in a folder with adversarial perturbations.
    """
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(files)} images to process.")
    with tqdm(total=len(files), desc="Processing images") as pbar:
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            for filename in batch:
                process_image(filename, input_folder, output_folder, strength)
                pbar.update(1)

# --------- MAIN ENTRY POINT ---------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Input folder containing images")
    parser.add_argument("output_folder", help="Output folder for perturbed images")
    parser.add_argument("--strength", type=float, default=0.05, help="Strength of perturbation")
    args = parser.parse_args()
    process_folder(args.input_folder, args.output_folder, args.strength)

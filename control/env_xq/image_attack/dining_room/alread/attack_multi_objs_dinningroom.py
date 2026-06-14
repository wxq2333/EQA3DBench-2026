import torch
import clip
from PIL import Image
import numpy as np
import sys
import concurrent.futures
import os
import logging
import shutil  # For file copying
sys.path.append('/home/ubuntu/桌面/control/env_xq/estool')
from es import OpenES  # Ensure OpenES module is correctly installed and path set
import yaml
import subprocess
import time, math, gc, pickle, random
from datetime import datetime
from operator import itemgetter
import warnings

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")

# Configure logging to output to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/image_diningroom/optimization_log.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# List of objects to optimize
object_names_to_update = [
    "banana",
    "flowers_in_vase",
    "banana(1)",
    "burger",
    "watermelon",
     "lub_sandwich_pile",
     "coffee_mug",
     "flowers_in_vase",
    "fruit_bowl"     # Add more objects here as needed
]

# Paths
yaml_path = "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/attack_diningroom/object_config_diningroom.yaml"
base_yaml_path = "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/attack_diningroom/object_config_diningroom_base.yaml"  # Baseline YAML file path

# Define multiple text descriptions
text_descriptions = [
    "Placing a vase on the dining table serves several purposes: Aesthetic Appeal",
    "From the images provided, there are a few items that are typically not considered appropriate for a dining room setting. Food items like a watermelon, a hamburger, and a banana",
    "The vase in the images appears to contain flowers, indicating that it is not                                                                  empty."
            ]

# Position and rotation ranges
position_ranges = (-1.34,-0.3, 1.0, -1, -3.1)  # (x_min, x_max, y_fixed, z_min, z_max)1
rotation_range = (-180, 180)  # (roll_min, roll_max)

# Normalization and denormalization functions for object parameters
def normalize_single_obj_params(params):
    x_min, x_max, y_fixed, z_min, z_max = position_ranges
    roll_min, roll_max = rotation_range

    # Handle division by zero
    x_range = x_max - x_min
    z_range = z_max - z_min
    if x_range == 0 or z_range == 0:
        logging.error("Normalization Error: Division by zero in position_ranges.")
        return [0.0, y_fixed, 0.0, 0.0, 0.0, 0.0]

    x_norm = (params[0] - x_min) / x_range
    z_norm = (params[2] - z_min) / z_range
    y = y_fixed

    roll_norm = (params[3] + 180) / 360
    pitch_norm = (params[4] + 180) / 360
    yaw_norm = (params[5] + 180) / 360

    # Ensure normalization within [0, 1]
    x_norm = np.clip(x_norm, 0.0, 1.0)
    z_norm = np.clip(z_norm, 0.0, 1.0)
    roll_norm = np.clip(roll_norm, 0.0, 1.0)
    pitch_norm = np.clip(pitch_norm, 0.0, 1.0)
    yaw_norm = np.clip(yaw_norm, 0.0, 1.0)

    return [x_norm, y, z_norm, roll_norm, pitch_norm, yaw_norm]

def denormalize_single_obj_params(normalized_params):
    x_min, x_max, y_fixed, z_min, z_max = position_ranges
    roll_min, roll_max = rotation_range

    x = normalized_params[0] * (x_max - x_min) + x_min
    z = normalized_params[2] * (z_max - z_min) + z_min
    y = y_fixed

    # Clamp positions to valid ranges
    x = np.clip(x, x_min, x_max)
    z = np.clip(z, z_min, z_max)

    roll = normalized_params[3] * 360 - 180
    pitch = normalized_params[4] * 360 - 180
    yaw = normalized_params[5] * 360 - 180

    # Ensure rotations are within [-180, 180]
    roll = ((roll + 180) % 360) - 180
    pitch = ((pitch + 180) % 360) - 180
    yaw = ((yaw + 180) % 360) - 180

    return [x, y, z, roll, pitch, yaw]


# Update YAML file with multiple objects' positions and rotations
def update_yaml_file_multiple(obj_names, positions_and_rotations_list, current_objects):
    for obj_name, pos_rot in zip(obj_names, positions_and_rotations_list):
        obj_index = next((i for i, obj in enumerate(current_objects) if obj['name'] == obj_name), None)
        if obj_index is not None:
            current_objects[obj_index]['position'] = [float(x) for x in pos_rot[:3]]
            current_objects[obj_index]['rotation'] = [float(r) for r in pos_rot[3:]]
        else:
            logging.warning(f"Object '{obj_name}' not found in YAML configuration.")

    with open(yaml_path, 'w') as f:
        yaml.dump({'objects': current_objects}, f, default_flow_style=False, allow_unicode=True)
    logging.info("Updated YAML with new positions and rotations for all specified objects.")

# Set objects' positions and rotations
def set_object_positions_and_rotations(obj_names, positions_and_rotations_list, current_objects):
    update_yaml_file_multiple(obj_names, positions_and_rotations_list, current_objects)

# Get scene images
def get_scene_images(max_retries=3, retry_delay=2):
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                ["python", "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/attack_diningroom/generate_three_images_dinningroom.py"],
                capture_output=True,
                text=True,  # Decode output as string
                check=True  # Raise exception if subprocess returns non-zero exit code
            )

            logging.info(f"get_scene_images: Attempt {attempt} successfully ran image generation script.")

            image_paths = [
                "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/generated_images/frame_000000.jpg",
                "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/generated_images/frame_000001.jpg",
                "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/generated_images/frame_000002.jpg"
            ]

            images = []
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    logging.error(f"get_scene_images: Image file does not exist: {image_path}")
                    raise FileNotFoundError(f"Image file not found: {image_path}")

                image = Image.open(image_path)
                image = image.convert('RGB')  # Ensure image is in RGB mode
                image_np = np.array(image)

                # Ensure image size is (512, 512, 3)
                expected_shape = (512, 512, 3)
                if image_np.shape != expected_shape:
                    logging.error(f"get_scene_images: Image shape mismatch: {image_path} - {image_np.shape}")
                    raise ValueError(f"Image shape mismatch: {image_np.shape}")

                # Log image size
                logging.info(f"get_scene_images: Image {image_path} size: {image_np.shape}")

                images.append(image_np)

            logging.info("get_scene_images: Successfully loaded all images.")
            return images

        except subprocess.CalledProcessError as cpe:
            logging.error(f"get_scene_images: Attempt {attempt} - Image generation script failed with error: {cpe.stderr}")
        except FileNotFoundError as fnfe:
            logging.error(f"get_scene_images: Attempt {attempt} - {fnfe}")
        except ValueError as ve:
            logging.error(f"get_scene_images: Attempt {attempt} - {ve}")
        except Exception as e:
            logging.error(f"get_scene_images: Attempt {attempt} - Unexpected error: {e}")

        if attempt < max_retries:
            logging.info(f"get_scene_images: Waiting {retry_delay} seconds before retrying...")
            time.sleep(retry_delay)

    logging.error("get_scene_images: All attempts failed.")
    return None

# Batch calculate image similarity
def batch_calculate_similarity(images, text_features):
    try:
        # Process images without adding extra dimensions
        image_tensors = torch.stack([preprocess(Image.fromarray(img)) for img in images]).to(device)
        image_tensors = image_tensors.type(torch.float32)  # Explicitly convert to float32

        # Log image tensor shape
        logging.info(f"batch_calculate_similarity: Image tensor shape: {image_tensors.shape}")

        # Ensure input is 4D tensor: [batch_size, channels, height, width]
        with torch.no_grad():
            image_features = model.encode_image(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize image features
            similarity = (image_features @ text_features.T).squeeze().cpu().numpy()  # Compute similarity

        logging.info(f"batch_calculate_similarity: Computed similarity values: {similarity}")

        return similarity
    except Exception as e:
        logging.error(f"batch_calculate_similarity: Exception occurred - {e}")
        return np.array([0.0] * len(images))  # Return default similarity values

# Fitness function
def fitness_func(normalized_params, text_features, current_objects):
    try:
        # Convert normalized parameters to actual positions and rotations
        positions_and_rotations_list = []
        for i in range(len(object_names_to_update)):
            start = i * 6
            end = start + 6
            single_obj_params = normalized_params[start:end]
            denorm_params = denormalize_single_obj_params(single_obj_params)
            positions_and_rotations_list.append(denorm_params)

        # Set objects' positions and rotations in YAML config
        set_object_positions_and_rotations(object_names_to_update, positions_and_rotations_list, current_objects)

        # Generate and get scene images
        images = get_scene_images()
        if images is None:
            # Log failure and return a large finite penalty value
            logging.error(f"Fitness Function: Unable to get images, params: {normalized_params}")
            return 1e6  # Large finite penalty value

        # Calculate similarity between images and precomputed text features
        similarities = batch_calculate_similarity(images, text_features)

        # Check if similarity values are finite numbers
        if not np.all(np.isfinite(similarities)):
            logging.error(f"Fitness Function: Encountered non-finite similarity values: {similarities}")
            return 1e6  # Large finite penalty value

        average_similarity = np.mean(similarities)
        logging.info(f"Fitness Function: Average similarity value: {average_similarity}")

        # Return negative similarity as fitness value (since ES is a minimization problem)
        return -average_similarity

    except Exception as e:
        # Catch any unexpected exceptions and return a penalty value
        logging.error(f"Fitness Function: Exception occurred - {e}")
        return 1e6  # Large finite penalty value

# Log positions and rotations for each generation
def log_positions_and_rotations(obj_names, positions_and_rotations_list, generation, text_description):
    # Create a directory to store logs if it doesn't exist
    log_dir = "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/images_diningroom/positions_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Define log file path with text description identifier
    sanitized_description = "_".join(text_description.split()[:5])  # Simple sanitization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")  # Include microseconds for uniqueness
    log_file_path = os.path.join(log_dir, f"positions_log_gen_{generation+1}_{sanitized_description}_{timestamp}.txt")
    
    try:
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Generation {generation+1}:\n")
            for obj_name, pos_rot in zip(obj_names, positions_and_rotations_list):
                position = pos_rot[:3]
                rotation = pos_rot[3:]
                log_file.write(f"Object: {obj_name}\n")
                log_file.write(f"Position: {position}\n")
                log_file.write(f"Rotation: {rotation}\n\n")
        logging.info(f"Logged positions and rotations for Generation {generation + 1}.")
    except Exception as e:
        logging.error(f"log_positions_and_rotations: Exception occurred - {e}")

# Parallel fitness evaluation
def parallel_fitness_evaluation(normalized_solutions, text_features, current_objects):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # Set max threads to 10
        futures = [
            executor.submit(fitness_func, sol, text_features, current_objects)
            for sol in normalized_solutions
        ]
        fitness_values = [future.result() for future in concurrent.futures.as_completed(futures)]
    return np.array(fitness_values)

# Optimization function for a single text description
def optimize_for_text(text_description, index):
    logging.info(f"Starting optimization for text description {index+1}: {text_description}")

    # Reload the baseline YAML to ensure a clean state
    try:
        with open(base_yaml_path, 'r') as f:
            base_data = yaml.safe_load(f)
            current_objects = base_data['objects']
        logging.info(f"Loaded baseline YAML for text description {index+1}.")
    except Exception as e:
        logging.error(f"Failed to load baseline YAML for text description {index+1}: {e}")
        return

    # Initialize OpenES within the optimization function to ensure a fresh state
    es = OpenES(
        num_params=len(object_names_to_update) * 6,  # Each object has 6 parameters (x, y, z, roll, pitch, yaw)
        sigma_init=0.1,
        popsize=50,  # Population size
        learning_rate=0.1,
        rank_fitness=False
    )

    # Tokenize text description
    text = clip.tokenize([text_description]).to(device)

    # Precompute text features
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    num_generations = 20  # Adjust number of generations as needed
    for generation in range(num_generations):
        logging.info(f"Text {index+1}: Starting Generation {generation+1}")
        normalized_solutions = es.ask()
        # Use parallel computation for fitness values
        fitness_values = parallel_fitness_evaluation(normalized_solutions, text_features, current_objects)
        es.tell(fitness_values)

        best_solution, best_fitness = es.result()[0], es.result()[1]

        logging.info(f"Text {index+1}, Generation {generation+1} - Best fitness: {best_fitness}")
        print(f"Text {index+1}, Generation {generation+1} - Best fitness: {best_fitness}")

        # Convert best solution to actual positions and rotations
        best_positions_and_rotations_list = []
        for i in range(len(object_names_to_update)):
            start = i * 6
            end = start + 6
            single_obj_params = best_solution[start:end]
            denorm_params = denormalize_single_obj_params(single_obj_params)
            best_positions_and_rotations_list.append(denorm_params)

        # Log positions and rotations for this generation
        log_positions_and_rotations(object_names_to_update, best_positions_and_rotations_list, generation, text_description)

        # Set objects' positions and rotations for the next generation
        set_object_positions_and_rotations(object_names_to_update, best_positions_and_rotations_list, current_objects)

        # Optionally, clear cache at the end of each generation to prevent memory leaks
        torch.cuda.empty_cache()
        gc.collect()

    logging.info(f"Optimization completed for text description {index+1}: {text_description}")

    # Save updated YAML file with comment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")  # Include microseconds for uniqueness
    sanitized_description = "_".join(text_description.split()[:5])  # Simple sanitization
    new_yaml_path = f"/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/attack_diningroom/object_config_diningroom_updated_{index+1}_{sanitized_description}_{timestamp}.yaml"
    comment = text_description  # Use text_description as the comment

    try:
        with open(new_yaml_path, 'w') as f:
            # Write comment
            f.write(f"# {comment}\n")
            # Write YAML data
            yaml.dump({'objects': current_objects}, f, default_flow_style=False, allow_unicode=True)
        logging.info(f"Saved updated YAML to {new_yaml_path} with comment.")
    except Exception as e:
        logging.error(f"Failed to save updated YAML file for text description {index+1}: {e}")

    # Restore the main YAML file from the baseline YAML file to ensure a clean state for the next optimization
    try:
        shutil.copy(base_yaml_path, yaml_path)
        logging.info(f"Restored {yaml_path} from {base_yaml_path} after optimizing for text description {index+1}.")
    except Exception as e:
        logging.error(f"Failed to restore {yaml_path} from {base_yaml_path} for text description {index+1}: {e}")

# Entry point for the script
def main():
    for idx, text_desc in enumerate(text_descriptions):
        optimize_for_text(text_desc, idx)

    logging.info("All optimizations completed.")

if __name__ == "__main__":
    main()

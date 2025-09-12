import requests
import os
from dotenv import load_dotenv, find_dotenv

from loguru import logger

# Load environment variables once at module load
env_path = find_dotenv()
load_dotenv(env_path)
BASE_URL = os.environ.get("FLASK_SERVER_URL")


def upload_images(person_img_path, cloth_img_path):
    url = f"{BASE_URL}/upload-images"
    files = {
        "person_image": open(person_img_path, "rb"),
        "cloth_image": open(cloth_img_path, "rb"),
    }
    headers = {"Accept": "application/json"}
    resp = requests.post(url, files=files, headers=headers)
    return resp.json()


def generate_masks():
    url = f"{BASE_URL}/generate-masks"
    headers = {"Accept": "application/json"}
    resp = requests.post(url, headers=headers)
    return resp.json()


def run_inference():
    url = f"{BASE_URL}/run-inference"
    headers = {"Accept": "application/json"}
    resp = requests.post(url, headers=headers)
    return resp.json()


def download_result_image(relative_path, save_path):
    url = f"{BASE_URL}/results/{relative_path}"
    resp = requests.get(url)
    if resp.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(resp.content)
        logger.info(f"Result image saved to {save_path}")
        return True
    else:
        logger.error(
            f"Failed to download result image. Status code: {resp.status_code}"
        )
        return False


def process_images_catvton(person_img_path, cloth_img_path, output_image_path):
    upload_resp = upload_images(person_img_path, cloth_img_path)
    if not upload_resp.get("success"):
        return {"success": False, "step": "upload", "response": upload_resp}

    mask_resp = generate_masks()
    if not mask_resp.get("success"):
        return {"success": False, "step": "mask_generation", "response": mask_resp}

    inference_resp = run_inference()
    if not inference_resp.get("success"):
        return {"success": False, "step": "inference", "response": inference_resp}

    result_image_path = inference_resp.get("result_image")
    if result_image_path:
        download_success = download_result_image(result_image_path, output_image_path)
        return {
            "success": download_success,
            "step": "download",
            "response": inference_resp,
        }
    else:
        return {
            "success": False,
            "step": "download",
            "response": "No result image path returned",
        }

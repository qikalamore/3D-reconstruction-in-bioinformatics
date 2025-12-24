import gradio as gr
import requests
import time
import json
from PIL import Image
from io import BytesIO
import os
import sys
import shutil
from datetime import datetime
import torch
import numpy as np
import glob
import gc
import re
from openai import OpenAI

# Add Depth Anything 3 to path
sys.path.append("/root/autodl-tmp/shengwuxxx/depth-anything-3/src")

try:
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.memory import cleanup_cuda_memory
    from depth_anything_3.utils.export.glb import export_to_glb
    from depth_anything_3.utils.export.gs import export_to_gs_video
except ImportError as e:
    print(f"Error importing Depth Anything 3 modules: {e}")
    print("Please ensure the 'depth-anything-3' directory is present and contains the necessary files.")
    sys.exit(1)

# --- ModelScope Configuration ---
API_KEY = "your ModelScope API key"  # replace with your ModelScope API key
BASE_URL = 'https://api-inference.modelscope.cn/'
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# --- Depth Anything 3 Model Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing Depth Anything 3 model on {device}...")
da3_model = None

def get_da3_model():
    global da3_model
    if da3_model is None:
        try:
            # Get model directory from environment variable or use default
            # User specified weights path: /root/autodl-tmp/shengwuxxx/depth-anything-3/weights
            model_dir = os.environ.get(
                "DA3_MODEL_DIR", "/root/autodl-tmp/shengwuxxx/depth-anything-3/weights"
            )
            print(f"Loading Depth Anything 3 model from {model_dir}...")
            da3_model = DepthAnything3.from_pretrained(model_dir)
            da3_model = da3_model.to(device)
            da3_model.eval()
            print("Depth Anything 3 Model loaded successfully.")
        except Exception as e:
            print(f"Error loading Depth Anything 3 model: {e}")
            da3_model = None
    return da3_model

# Initialize model at startup
get_da3_model()

# --- Helper Functions ---

def upload_image_to_get_url(file_path):
    url = 'https://img.scdn.io/api/v1.php'
    try:
        with open(file_path, 'rb') as f:
            files = {'image': f}
            data = {'cdn_domain': 'img.scdn.io'}
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            print("图片上传成功，获取URL为", response.json()['url'])
            return response.json()['url']
    except Exception as e:
        print(f"Failed to upload image: {e}")
        return None

def check_is_medical(image_url):
    client = OpenAI(
        base_url='https://api-inference.modelscope.cn/v1',
        api_key=API_KEY,
    )

    try:
        response = client.chat.completions.create(
            model='Qwen/Qwen3-VL-8B-Instruct',
            messages=[{
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': '这张图是否严格和医学有关？请你回答，你只需要回答是或者不是',
                }, {
                    'type': 'image_url',
                    'image_url': {
                        'url': image_url,
                    },
                }],
            }],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Medical check failed: {e}")
        # In case of API error, we might want to allow it to proceed or fail. 
        # Here we return Error to let the caller decide, or just return empty string to pass.
        # Let's return "Error" and log it.
        return "Error"

def handle_local_upload(files, progress=gr.Progress()):
    gr.Info("本地上传模式：直接上传多张已有图片用于重建，不进行AI生成，如果你对图像很有信心，请使用它！", duration=5)
    if not files:
        raise gr.Error("请至少上传一张图片。")
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(os.path.dirname(__file__), "generated_data", f"session_upload_{timestamp}")
    images_dir = os.path.join(session_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    saved_files = []
    progress(0, desc="正在处理上传...")
    for i, file in enumerate(files):
        # Gradio passes file objects with .name as path
        file_path = file.name
        filename = os.path.basename(file_path)
        dest_path = os.path.join(images_dir, filename)
        shutil.copy(file_path, dest_path)
        saved_files.append(dest_path)
        
    return session_dir, saved_files, f"✅ 已上传 {len(saved_files)} 张图片。准备重建。", gr.update(interactive=True)

def translate_prompt(text):
    if not text:
        return text
    
    # Check if text contains Chinese characters
    if re.search(r'[\u4e00-\u9fff]', text):
        print(f"检测到中文提示词: {text}，正在翻译...")
        gr.Info(f"检测到中文提示词，正在翻译: {text}")
        
        client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key=API_KEY,
        )
        
        # set extra_body for thinking control
        extra_body = {
            # enable thinking, set to False to disable test
            "enable_thinking": True,
            # use thinking_budget to contorl num of tokens used for thinking
            # "thinking_budget": 4096
        }
        
        try:
            response = client.chat.completions.create(
                model='Qwen/Qwen3-8B',
                messages=[
                    {
                        'role': 'user',
                        'content': f'严格将以下句子翻译成英文，不需要任何额外内容: "{text}"'
                    }
                ],
                stream=True,
                extra_body=extra_body
            )
            
            translated_text = ""
            for chunk in response:
                if chunk.choices:
                    answer_chunk = chunk.choices[0].delta.content
                    if answer_chunk:
                        translated_text += answer_chunk
            
            translated_text = translated_text.strip()
            # Remove quotes if present in the output (sometimes models add them)
            if translated_text.startswith('"') and translated_text.endswith('"'):
                translated_text = translated_text[1:-1]
            
            print(f"翻译结果: {translated_text}")
            gr.Info(f"翻译完成: {translated_text}")
            return translated_text
        except Exception as e:
            print(f"Translation failed: {e}")
            return text
    return text

def generate_multiview_images(image_url, additional_prompt="", progress=gr.Progress()):
    if not image_url:
        return None, None, "请输入图片 URL。"

    # Translate prompt if needed
    additional_prompt = translate_prompt(additional_prompt)

    # Medical Check
    # progress(0, desc="正在检测图片是否与医学相关...")
    answer = check_is_medical(image_url)
    print(f"Medical check result: {answer}")
    
    if answer == "Error":
        return None, None, "医学检测失败（API 错误）大概率为连接问题，请稍后重试。"

    negative_keywords = ["不是", "否", "no", "No", "NO"]
    # Check if any negative keyword is in the answer
    if any(keyword in answer for keyword in negative_keywords):
        return None, None, "MEDICAL_ERROR"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a session directory
    session_dir = os.path.join(os.path.dirname(__file__), "generated_data", f"session_{timestamp}")
    images_dir = os.path.join(session_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Views configuration
    views = ["top view", "left side view", "right side view", "bottom view", "back view"]
    view_trans = {
        "top view": "顶视图", "left side view": "左视图", "right side view": "右视图", 
        "bottom view": "底视图", "back view": "后视图"
    }
    view_filename_map = {
        "top view": "top_view.jpg",
        "left side view": "left_view.jpg",
        "right side view": "right_view.jpg",
        "bottom view": "bottom_view.jpg",
        "back view": "back_view.jpg"
    }
    
    generated_files = []
    
    # 1. Generate Front View (White Background)
    progress(0, desc="正在生成白底正视图...")
    front_prompt = f"Generate the front view of the object in the image. The image must have a pure white background and contain strictly a single object. Ensure the object is identical to the original in terms of morphology, geometry, and texture. Photorealistic, full color, high resolution, rich texture details, accurate lighting consistent with a front-facing perspective. {additional_prompt}"
    
    try:
        # Submit Task for Front View
        payload = {
            "model": 'Qwen/Qwen-Image-Edit-2509',
            "prompt": front_prompt,
            "image_url": [image_url]
        }
        response = requests.post(
            f"{BASE_URL}v1/images/generations",
            headers={**HEADERS, "X-ModelScope-Async-Mode": "true"},
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8')
        )
        response.raise_for_status()
        task_id = response.json()["task_id"]
        
        # Poll for result
        generated_front_url = None
        while True:
            result = requests.get(
                f"{BASE_URL}v1/tasks/{task_id}",
                headers={**HEADERS, "X-ModelScope-Task-Type": "image_generation"},
            )
            result.raise_for_status()
            data = result.json()
            
            if data["task_status"] == "SUCCEED":
                generated_front_url = data["output_images"][0]
                break
            elif data["task_status"] == "FAILED":
                print(f"Failed to generate front view: {data}")
                return None, None, "GENERATION_FAILED:正视图"
            
            time.sleep(2)
            
        # Download and save the generated front view
        response = requests.get(generated_front_url)
        response.raise_for_status()
        front_img = Image.open(BytesIO(response.content))
        if front_img.mode == 'RGBA':
            front_img = front_img.convert('RGB')
            
        front_path = os.path.join(images_dir, "front_view.jpg")
        front_img.save(front_path)
        generated_files.append(front_path)
        
        # Update image_url to use the generated one for subsequent views
        image_url = generated_front_url
        
    except Exception as e:
        print(f"Error generating front view: {e}")
        return None, None, f"GENERATION_ERROR:正视图:{str(e)}"

    # 2. Generate Other Views
    view_prompts = {
        "top view": "Generate the top view of the object in the image, as seen directly from above. The image must have a pure white background and contain strictly a single object. Focus on the upper surface texture, color, and geometry. Ensure the object's proportions and morphology are consistent with the original object. Photorealistic, full color, high resolution, detailed top-down perspective.",
        "left side view": "Generate the left side view of the object in the image, showing the object's profile from the left. The image must have a pure white background and contain strictly a single object. Focus on the side geometry, depth, and texture details. Ensure the object is identical to the original in color and material. Photorealistic, full color, accurate side profile.",
        "right side view": "Generate the right side view of the object in the image, showing the object's profile from the right. The image must have a pure white background and contain strictly a single object. Focus on the side geometry, depth, and texture details. Ensure the object is identical to the original in color and material. Photorealistic, full color, accurate side profile.",
        "bottom view": "Generate the bottom view of the object in the image, as seen directly from below. The image must have a pure white background and contain strictly a single object. Focus on the base texture, color, and structure. Ensure the object's proportions and morphology are consistent with the original object. Photorealistic, full color, detailed bottom-up perspective.",
        "back view": "Generate the back view of the object in the image, showing the rear side. The image must have a pure white background and contain strictly a single object. Focus on the rear texture, color, and geometry details. Ensure the object is identical to the original. Photorealistic, full color, consistent with the front view's scale and style."
    }

    for i, view in enumerate(views):
        view_name_cn = view_trans.get(view, view)
        progress((i + 1) / (len(views) + 1), desc=f"正在生成 {view_name_cn}...")
        
        base_prompt = view_prompts.get(view, f"Generate the {view} of the object in the image, pure white background, strictly single object, identical to the original object, high quality")
        prompt = f"{base_prompt} {additional_prompt}"
        
        try:
            # Submit Task
            payload = {
                "model": 'Qwen/Qwen-Image-Edit-2509',
                "prompt": prompt,
                "image_url": [image_url]
            }
            response = requests.post(
                f"{BASE_URL}v1/images/generations",
                headers={**HEADERS, "X-ModelScope-Async-Mode": "true"},
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8')
            )
            response.raise_for_status()
            task_id = response.json()["task_id"]
            
            # Poll for result
            while True:
                result = requests.get(
                    f"{BASE_URL}v1/tasks/{task_id}",
                    headers={**HEADERS, "X-ModelScope-Task-Type": "image_generation"},
                )
                result.raise_for_status()
                data = result.json()
                
                if data["task_status"] == "SUCCEED":
                    output_url = data["output_images"][0]
                    img_data = requests.get(output_url).content
                    image = Image.open(BytesIO(img_data))
                    
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    
                    filename = view_filename_map[view]
                    save_path = os.path.join(images_dir, filename)
                    image.save(save_path)
                    generated_files.append(save_path)
                    break
                elif data["task_status"] == "FAILED":
                    print(f"Failed to generate {view}: {data}")
                    return None, None, f"GENERATION_FAILED:{view_name_cn}"
                
                time.sleep(2)
                
        except Exception as e:
            print(f"Error generating {view}: {e}")
            return None, None, f"GENERATION_ERROR:{view_name_cn}:{str(e)}"
            
    # Return the session directory (parent of 'images') and list of files
    return session_dir, generated_files, "✅ 生成完成！现在可以重建 3D 模型。"

def run_depth_anything_inference(target_dir, conf_thres, show_cam, mask_sky, mask_black_bg, mask_white_bg, prediction_mode, 
                                 process_res_method, infer_gs, ref_view_strategy, num_max_points, gs_trj_mode, gs_video_quality,
                                 progress=gr.Progress()):
    model = get_da3_model()
    if model is None:
        raise gr.Error("Depth Anything 3 模型未加载。")
        
    if not target_dir or not os.path.exists(target_dir):
        raise gr.Error("未找到目标目录。请先生成图片。")
        
    print(f"Processing images from {target_dir}")
    progress(0.1, desc="正在加载图片...")
    
    # Prepare images
    image_folder_path = os.path.join(target_dir, "images")
    all_image_paths = sorted(glob.glob(os.path.join(image_folder_path, "*")))
    
    # Filter for image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    image_paths = [
        path
        for path in all_image_paths
        if any(path.lower().endswith(ext) for ext in image_extensions)
    ]
    
    if len(image_paths) == 0:
        raise gr.Error("目标目录中未找到图片。")

    progress(0.3, desc="正在运行 Depth Anything 3 推理...")
    
    # Inference
    try:
        # Map UI options
        save_percentage = conf_thres
        filter_black_bg = mask_black_bg
        filter_white_bg = mask_white_bg
        
        # Map process_res_method to actual method name (logic from model_inference.py)
        method_mapping = {"high_res": "lower_bound_resize", "low_res": "upper_bound_resize"}
        actual_method = method_mapping.get(process_res_method, "upper_bound_crop")
        
        with torch.no_grad():
            prediction = model.inference(
                image_paths,
                export_dir=None,
                process_res_method=actual_method,
                infer_gs=infer_gs,
                ref_view_strategy=ref_view_strategy,
            )
            
        progress(0.8, desc="正在生成 3D 模型...")
        
        # Export to GLB
        export_to_glb(
            prediction,
            filter_black_bg=filter_black_bg,
            filter_white_bg=filter_white_bg,
            export_dir=target_dir,
            show_cameras=show_cam,
            conf_thresh_percentile=save_percentage,
            num_max_points=int(num_max_points * 1000), # Convert K to actual count
        )
        
        # Export to GS Video if enabled
        gsvideo_path = None
        if infer_gs:
            progress(0.9, desc="正在生成 3DGS 视频...")
            mode_mapping = {"extend": "extend", "smooth": "interpolate_smooth"}
            try:
                export_to_gs_video(
                    prediction,
                    export_dir=target_dir,
                    chunk_size=4,
                    trj_mode=mode_mapping.get(gs_trj_mode, "extend"),
                    enable_tqdm=True,
                    vis_depth=None,
                    video_quality=gs_video_quality,
                )
                gsvideo_path = sorted(glob.glob(os.path.join(target_dir, "gs_video", "*.mp4")))[-1]
            except Exception as e:
                print(f"Error generating GS video: {e}")
        
        glb_path = os.path.join(target_dir, "scene.glb")
        
        # Save predictions cache (logic from model_inference.py)
        output_file = os.path.join(target_dir, "predictions.npz")
        save_dict = {}
        if prediction.processed_images is not None:
            save_dict["images"] = prediction.processed_images
        if prediction.depth is not None:
            save_dict["depths"] = np.round(prediction.depth, 6)
        if prediction.conf is not None:
            save_dict["conf"] = np.round(prediction.conf, 2)
        if prediction.extrinsics is not None:
            save_dict["extrinsics"] = prediction.extrinsics
        if prediction.intrinsics is not None:
            save_dict["intrinsics"] = prediction.intrinsics
        np.savez_compressed(output_file, **save_dict)
        
        cleanup_cuda_memory()
        
        progress(1.0, desc="完成！")
        return glb_path, gsvideo_path
        
    except Exception as e:
        cleanup_cuda_memory()
        raise gr.Error(f"推理出错: {e}")

def update_visualization(target_dir, conf_thres, show_cam, mask_sky, mask_black_bg, mask_white_bg, prediction_mode, num_max_points):
    if not target_dir:
        return None, None
    
    pred_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(pred_path):
        return None, None
        
    try:
        predictions = np.load(pred_path, allow_pickle=True)
        
        class PredictionWrapper:
            def __init__(self, data):
                self.processed_images = data.get("images")
                self.depth = data.get("depths")
                self.conf = data.get("conf")
                self.extrinsics = data.get("extrinsics")
                self.intrinsics = data.get("intrinsics")
        
        pred_wrapper = PredictionWrapper(predictions)
        
        # Export to GLB
        export_to_glb(
            pred_wrapper,
            filter_black_bg=mask_black_bg,
            filter_white_bg=mask_white_bg,
            export_dir=target_dir,
            show_cameras=show_cam,
            conf_thresh_percentile=conf_thres,
            num_max_points=int(num_max_points * 1000),
        )
        
        generated_glb = os.path.join(target_dir, "scene.glb")
        glb_path = os.path.join(target_dir, f"viz_{time.time()}.glb")
        
        if os.path.exists(generated_glb):
             shutil.copy(generated_glb, glb_path)
             return glb_path, gr.update(value=glb_path, visible=True)
        else:
             return None, None
    except Exception as e:
        print(f"Visualization update failed: {e}")
        return None, None

# --- Gradio UI ---
# Light Bio-tech Theme
theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="emerald",
    neutral_hue="slate",
).set(
    body_background_fill="#f0f9ff", # Light blue-ish white
    body_text_color="#0f172a",      # Dark slate
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e2e8f0",
    block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1)",
    button_primary_background_fill="linear-gradient(90deg, #06b6d4, #10b981)",
    button_primary_background_fill_hover="linear-gradient(90deg, #0891b2, #059669)",
    button_primary_text_color="#ffffff",
    button_primary_border_color="#22d3ee",
    input_background_fill="#f8fafc",
    input_border_color="#cbd5e1",
    input_placeholder_color="#94a3b8",
)

css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');

:root {
    /* Light Theme Variables (Default) */
    --custom-body-bg: #f0f9ff;
    --custom-body-bg-img: radial-gradient(circle at 50% 50%, #ffffff 0%, #f0f9ff 100%), linear-gradient(0deg, rgba(6, 182, 212, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(6, 182, 212, 0.03) 1px, transparent 1px);
    --custom-header-bg: rgba(255, 255, 255, 0.8);
    --custom-header-border: #cffafe;
    --custom-header-shadow: 0 10px 15px -3px rgba(6, 182, 212, 0.1);
    --custom-header-h1: #0e7490;
    --custom-header-p: #475569;
    --custom-panel-bg: rgba(255, 255, 255, 0.9);
    --custom-panel-border: #e2e8f0;
    --custom-panel-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    --custom-scrollbar-track: #f1f5f9;
    --custom-scrollbar-thumb: #cbd5e1;
    --custom-scrollbar-thumb-hover: #94a3b8;
    --custom-tab-text: #64748b;
    --custom-tab-border: #e2e8f0;
    --custom-status-bg: rgba(16, 185, 129, 0.1);
    --custom-status-color: #059669;
    --custom-status-border: #059669;
}

body.dark-theme {
    /* Dark Theme Variables */
    --custom-body-bg: #000000;
    --custom-body-bg-img: radial-gradient(circle at 50% 50%, #111827 0%, #000000 100%), linear-gradient(0deg, rgba(6, 182, 212, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(6, 182, 212, 0.05) 1px, transparent 1px);
    --custom-header-bg: rgba(15, 23, 42, 0.6);
    --custom-header-border: #06b6d4;
    --custom-header-shadow: 0 0 20px rgba(6, 182, 212, 0.2);
    --custom-header-h1: #22d3ee;
    --custom-header-p: #cbd5e1; /* Lighter gray for better visibility */
    --custom-panel-bg: rgba(15, 23, 42, 0.8);
    --custom-panel-border: #1e293b;
    --custom-panel-shadow: 0 0 15px rgba(6, 182, 212, 0.1);
    --custom-scrollbar-track: #0f172a;
    --custom-scrollbar-thumb: #1e293b;
    --custom-scrollbar-thumb-hover: #334155;
    --custom-tab-text: #cbd5e1; /* Lighter gray */
    --custom-tab-border: #1e293b;
    --custom-status-bg: rgba(16, 185, 129, 0.1);
    --custom-status-color: #10b981;
    --custom-status-border: #10b981;

    /* Gradio Overrides for Dark Mode */
    --body-background-fill: #050505;
    --body-text-color: #f1f5f9; /* Very light gray/white */
    --body-text-color-subdued: #cbd5e1;
    --block-background-fill: #0f172a;
    --block-border-color: #1e293b;
    --block-label-text-color: #e2e8f0;
    --block-title-text-color: #f8fafc;
    --input-background-fill: #1e293b; /* Slightly lighter than block bg */
    --input-border-color: #334155;
    --input-placeholder-color: #94a3b8;
    --input-text-color: #f8fafc;
    --prose-text-color: #e2e8f0;
    --prose-header-text-color: #f1f5f9;
    --table-text-color: #e2e8f0;
}

/* Force text color in dark mode for specific elements that might be stubborn */
body.dark-theme .gradio-container label, 
body.dark-theme .gradio-container span, 
body.dark-theme .gradio-container p,
body.dark-theme .gradio-container h1,
body.dark-theme .gradio-container h2,
body.dark-theme .gradio-container h3,
body.dark-theme .gradio-container h4,
body.dark-theme .gradio-container h5,
body.dark-theme .gradio-container h6 {
    color: #e2e8f0;
}

body.dark-theme .header h1 {
    /* Keep the gradient for the main header */
    background: linear-gradient(to right, #06b6d4, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent !important;
}

body.dark-theme .header p {
    color: #cbd5e1 !important;
}

body {
    font-family: 'Rajdhani', sans-serif !important;
    background-color: var(--custom-body-bg);
    background-image: var(--custom-body-bg-img);
    background-size: 100% 100%, 40px 40px, 40px 40px;
    transition: background 0.3s ease;
}

.container { 
    max-width: 95%; 
    margin: auto; 
    padding: 20px;
}

.header { 
    text-align: center; 
    margin-bottom: 30px; 
    padding: 20px;
    background: var(--custom-header-bg);
    border: 1px solid var(--custom-header-border);
    border-radius: 15px;
    box-shadow: var(--custom-header-shadow);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.header h1 { 
    font-family: 'Orbitron', sans-serif;
    font-size: 3em !important; 
    color: var(--custom-header-h1);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

.header p { 
    font-size: 1.2em; 
    color: var(--custom-header-p); 
    font-family: 'Rajdhani', sans-serif;
    letter-spacing: 1px;
}

.panel-container {
    background: var(--custom-panel-bg);
    border: 1px solid var(--custom-panel-border);
    border-radius: 15px;
    padding: 20px;
    box-shadow: var(--custom-panel-shadow);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.panel-container:hover {
    border-color: #06b6d4;
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.2);
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
::-webkit-scrollbar-track {
    background: var(--custom-scrollbar-track); 
}
::-webkit-scrollbar-thumb {
    background: var(--custom-scrollbar-thumb); 
    border-radius: 5px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--custom-scrollbar-thumb-hover); 
}

/* Button Glow Effects */
button.primary {
    box-shadow: 0 4px 6px -1px rgba(6, 182, 212, 0.3) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    font-weight: 700 !important;
    letter-spacing: 1px;
}
button.primary:hover {
    box-shadow: 0 10px 15px -3px rgba(6, 182, 212, 0.5) !important;
    transform: translateY(-2px);
}

/* Status Text */
#status {
    font-family: 'Orbitron', sans-serif;
    color: var(--custom-status-color);
    font-size: 1.1em;
    padding: 10px;
    border-left: 3px solid var(--custom-status-border);
    background: var(--custom-status-bg);
    margin-top: 10px;
}

/* Hide progress bar in status output */
#status .progress-text, 
#status .progress-level, 
#status .loading,
#status .meta-text {
    display: none !important;
}

/* Hide download button in Model3D */
.gradio-model3d a[download] { display: none !important; }
.gradio-model3d button[aria-label="Download"] { display: none !important; }
.gradio-model3d .download { display: none !important; }

/* Tabs Styling */
.tabs {
    border-bottom: 1px solid var(--custom-tab-border);
}
.tab-nav button {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--custom-tab-text);
}
.tab-nav button.selected {
    color: #06b6d4 !important;
    border-bottom: 2px solid #06b6d4 !important;
}

/* Theme Toggle Button */
#theme-toggle {
    position: absolute;
    top: 20px;
    right: 5px;
    z-index: 999;
    font-size: 1.5em;
    background: transparent;
    border: none;
    cursor: pointer;
    padding: 5px;
    border-radius: 50%;
    transition: transform 0.3s ease;
    width: 50px;
    height: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
}
#theme-toggle:hover {
    transform: rotate(15deg) scale(1.1);
    background: rgba(128,128,128,0.1);
}
"""

with gr.Blocks(theme=theme, css=css, title="Bio-Tech 3D Reconstruction") as demo:
    # Theme Toggle Button
    theme_toggle_btn = gr.Button("☀️", elem_id="theme-toggle")
    
    # JavaScript to toggle theme
    demo.load(
        None,
        None,
        None,
        js="""
        () => {
            const btn = document.getElementById('theme-toggle');
            if (btn) {
                btn.addEventListener('click', () => {
                    document.body.classList.toggle('dark-theme');
                    if (document.body.classList.contains('dark-theme')) {
                        btn.innerText = "🌙";
                    } else {
                        btn.innerText = "☀️";
                    }
                });
            }
        }
        """
    )

    with gr.Column(elem_classes="container"):
        with gr.Column(elem_classes="header"):
            gr.Markdown("# 🧬 Bio-Medical 3D Reconstruction Core")
            gr.Markdown("### 智能医学影像三维重建系统 | Intelligent Medical Image 3D Reconstruction System")
        
        target_dir_state = gr.State()
        
        with gr.Row(equal_height=False):
            # Left Column: Generation
            with gr.Column(scale=4, elem_classes="panel-container"):
                gr.Markdown("### 🔬 影像输入与生成 | Image Input & Generation")
                gr.Markdown("---")
                
                with gr.Tabs():
                    with gr.TabItem("🖼️ 图像上传"):
                        image_url_input = gr.Image(
                            label="上传医学影像 / Upload Medical Image", 
                            type="filepath",
                            elem_id="upload_img"
                        )
                        with gr.Accordion("⚙️ 高级选项 | Advanced Options", open=False):
                            with gr.Row():
                                use_additional_prompt = gr.Checkbox(label="启用增强提示词", value=False)
                                additional_prompt_input = gr.Textbox(label="提示词", placeholder="输入额外的生物特征描述", visible=False, lines=3)
                        
                        def toggle_prompt(checkbox):
                            return gr.update(visible=checkbox)
                        
                        use_additional_prompt.change(toggle_prompt, inputs=use_additional_prompt, outputs=additional_prompt_input)

                        with gr.Row():
                            generate_btn = gr.Button("🧬 启动多视角生成 | GENERATE", variant="primary")
                            clear_btn = gr.Button("🔄 重置系统 | RESET", variant="secondary", interactive=False)

                    with gr.TabItem("🔗 URL 输入"):
                        manual_url_input = gr.Textbox(
                            label="影像 URL / Image URL", 
                            placeholder="https://example.com/scan.jpg",
                            info="请输入医学影像的直接链接。"
                        )
                        with gr.Accordion("⚙️ 高级选项 | Advanced Options", open=False):
                            with gr.Row():
                                use_additional_prompt_url = gr.Checkbox(label="启用增强提示词", value=False)
                                additional_prompt_input_url = gr.Textbox(label="提示词", placeholder="输入额外的生物特征描述", visible=False, lines=3)
                        
                        use_additional_prompt_url.change(toggle_prompt, inputs=use_additional_prompt_url, outputs=additional_prompt_input_url)

                        with gr.Row():
                            generate_url_btn = gr.Button("🧬 启动多视角生成 (URL) | GENERATE", variant="primary")
                            clear_url_btn = gr.Button("🔄 重置系统 | RESET", variant="secondary", interactive=False)
                    
                    with gr.TabItem("📂 批量上传"):
                        image_upload_input = gr.File(
                            file_count="multiple", 
                            label="批量上传影像序列",
                            file_types=["image"]
                        )
                        upload_btn = gr.Button("📥 加载影像序列 | LOAD SEQUENCE", variant="primary")

                status_output = gr.Markdown("✅ 系统就绪 | SYSTEM READY", elem_id="status")
                gallery = gr.Gallery(
                    label="多视角序列 | Multi-view Sequence", 
                    columns=3, 
                    height="auto",
                    object_fit="contain",
                    show_label=True,
                    elem_id="gallery"
                )
                
            # Right Column: Reconstruction
            with gr.Column(scale=6, elem_classes="panel-container"):
                gr.Markdown("### 🧊 三维全息重建 | Holographic 3D Reconstruction")
                gr.Markdown("---")
                
                countdown_text = gr.Markdown("", elem_id="countdown")
                reconstruct_btn = gr.Button("🏗️ 启动三维重建引擎 | INITIATE RECONSTRUCTION", variant="primary", interactive=False)
                
                with gr.Tabs():
                    with gr.Tab("3D 模型预览 | 3D Model Preview"):
                        model_output = gr.Model3D(
                            label="3D 模型预览 | 3D Model Preview", 
                            height=600,
                            camera_position=(90, 90, 3), # Optional initial camera pos
                            interactive=False,
                            elem_id="model3d"
                        )
                    with gr.Tab("3DGS 渲染视频 | 3DGS Rendered Video"):
                        with gr.Column():
                            gs_info = gr.Markdown(
                                (
                                    "‼️ **3D Gaussian Splatting 渲染当前已禁用。** <br><br><br>"
                                    "要从 3DGS 渲染新视角，请在下方启用 **推断 3D Gaussian Splatting**。<br>"
                                    "然后，在 **3DGS 设置** 中，"
                                    "*可选* 配置 **渲染轨迹** (默认: smooth) "
                                    "和 **视频质量** (默认: low)，"
                                    "最后点击 **启动三维重建引擎**。"
                                ),
                                visible=True,
                                height=600,
                            )
                            gs_video = gr.Video(
                                height=600,
                                label="3DGS 渲染新视角视频 (深度仅供参考)",
                                interactive=False,
                                visible=False,
                            )
                
                download_model_btn = gr.DownloadButton("💾 导出模型数据 | EXPORT MODEL", elem_classes="rounded-button", visible=False)
                
                with gr.Accordion("🛠️ 开发者控制台 | Developer Console", open=False):
                    # Hidden controls
                    show_cam = gr.Checkbox(label="显示相机", value=True, visible=True)
                    mask_sky = gr.Checkbox(value=False, visible=False) # Deprecated for DA3
                    mask_black_bg = gr.Checkbox(label="过滤黑色背景", value=False, visible=True)
                    mask_white_bg = gr.Checkbox(label="过滤白色背景", value=False, visible=True)
                    
                    prediction_mode = gr.Radio(
                        choices=[("深度图与相机分支", "Depthmap and Camera Branch"), ("点云图分支", "Pointmap Branch")],
                        value="Depthmap and Camera Branch",
                        label="预测算法模式",
                        visible=False # Not used in DA3 directly in this way
                    )
                    
                    gr.Markdown("### 推理设置")
                    with gr.Row():
                        process_res_method_dropdown = gr.Dropdown(
                            choices=["high_res", "low_res"],
                            value="low_res",
                            label="图像处理方法",
                            info="low_res 适用于更多图片",
                            scale=1,
                        )
                        infer_gs = gr.Checkbox(
                            label="推断 3D Gaussian Splatting",
                            value=False,
                            info='启用 3DGS 新视角渲染（需要额外处理时间）',
                            scale=1,
                        )
                        ref_view_strategy_dropdown = gr.Dropdown(
                            choices=["saddle_balanced", "saddle_sim_range", "first", "middle"],
                            value="saddle_balanced",
                            label="参考视图策略",
                            info="从多个输入中选择参考视图的策略",
                            scale=1,
                        )
                    
                    gr.Markdown("### 3DGS 设置")
                    with gr.Row():
                        gs_trj_mode = gr.Dropdown(
                            choices=["smooth", "extend"],
                            value="smooth",
                            label="3DGS 渲染轨迹",
                            info="'smooth' 用于视图插值；'extend' 用于更长的轨迹",
                            visible=False,
                        )
                        gs_video_quality = gr.Dropdown(
                            choices=["low", "medium", "high"],
                            value="low",
                            label="3DGS 视频质量",
                            info="'low' 加载速度更快；'high' 视觉质量更好",
                            visible=False,
                        )
                        
                    # bind visibility of gs_trj_mode to infer_gs
                    infer_gs.change(
                        fn=lambda checked: (
                            gr.update(visible=checked),
                            gr.update(visible=checked),
                            gr.update(visible=checked),
                            gr.update(visible=(not checked)),
                        ),
                        inputs=infer_gs,
                        outputs=[gs_trj_mode, gs_video_quality, gs_video, gs_info],
                    )
                    
                    gr.Markdown("### 可视化设置")
                    num_max_points = gr.Slider(
                        minimum=1000,
                        maximum=100000,
                        value=1000,
                        step=1000,
                        label="最大点数 (千点)",
                        info="导出到 GLB 的最大点数（以千为单位）",
                    )
                
                # Confidence Threshold Slider (Visible)
                conf_thres = gr.Slider(0, 100, value=10, label="置信度过滤 | Confidence Threshold (%)", info="数值越高，过滤的点越多。", visible=True)
                
                update_btn = gr.Button("🔄 更新视图 | UPDATE VIEW", variant="secondary", visible=False)

    # Event Handlers
    def on_generate_click(file_path, use_prompt, prompt_text):
        gr.Info("图片生成模式：上传单张图片，AI自动生成多视角图像用于重建。", duration=5)
        if not file_path:
             gr.Warning("请先上传图片。")
             yield (
                gr.State(), None, "请先上传图片。", gr.update(interactive=False), None, gr.update(interactive=True), gr.update(interactive=False)
             )
             return

        # 1. Disable generate, disable clear
        yield (
            gr.State(), # target_dir (no change yet)
            None,       # gallery
            gr.update(), # status
            gr.update(interactive=False), # reconstruct_btn
            file_path,        # url input
            gr.update(interactive=False), # generate_btn
            gr.update(interactive=False)  # clear_btn
        )
        
        url = upload_image_to_get_url(file_path)
        if not url:
             gr.Warning("图片上传失败。")
             yield (
                gr.State(), None, "图片上传失败。", gr.update(interactive=False), file_path, gr.update(interactive=True), gr.update(interactive=False)
             )
             return

        yield (
            gr.State(), None, gr.update(), gr.update(interactive=False), file_path, gr.update(interactive=False), gr.update(interactive=False)
        )
        
        # 2. Run generation
        session_dir, files, msg = generate_multiview_images(url, prompt_text if use_prompt else "")
        
        # 3. Handle result
        if "FAILED" in str(msg) or "ERROR" in str(msg) or "失败" in str(msg) or "无关" in str(msg) or "请输入" in str(msg):
             # Failure case
             if msg == "MEDICAL_ERROR":
                 gr.Warning("您上传的图片和医学无关")
                 msg = "就绪。"
                 url_out = None
             elif msg == "Medical check failed (API Error).":
                 gr.Warning("医学检测失败（API 错误）请稍后重试。")
                 msg = "就绪。"
                 url_out = file_path
             elif msg == "请输入图片 URL。":
                 gr.Warning(msg)
                 msg = "就绪。"
                 url_out = file_path
             elif msg and str(msg).startswith("GENERATION_FAILED:"):
                 view_name = msg.split(":")[1]
                 gr.Warning(f"{view_name} 生成失败。请重试。")
                 msg = f"生成 {view_name} 失败。"
                 url_out = file_path
             elif msg and str(msg).startswith("GENERATION_ERROR:"):
                 parts = msg.split(":", 2)
                 view_name = parts[1]
                 error_detail = parts[2] if len(parts) > 2 else "未知错误"
                 gr.Warning(f"生成 {view_name} 出错: {error_detail}")
                 msg = f"生成 {view_name} 出错。"
                 url_out = file_path
             else:
                 url_out = file_path

             yield (
                None, 
                None, 
                msg, 
                gr.update(interactive=False), 
                url_out,
                gr.update(interactive=True),  # Re-enable generate
                gr.update(interactive=False)  # Keep clear disabled
             )
        else:
             # Success case
             yield (
                session_dir,
                files,
                msg,
                gr.update(interactive=True), # Enable reconstruct
                file_path,
                gr.update(interactive=False), # Keep generate DISABLED
                gr.update(interactive=True)   # Enable clear
             )

    def on_clear_click():
        return (
            None, # target_dir
            None, # gallery
            "已清空。准备生成。", # status
            gr.update(interactive=False), # reconstruct_btn
            None,   # url input
            gr.update(interactive=True),  # Enable generate
            gr.update(interactive=False)  # Disable clear
        )

    def on_generate_url_click(url, use_prompt, prompt_text):
        gr.Info("URL生成模式：输入图片链接，AI自动生成多视角图像用于重建。当图片URL-API无法连接时建议使用", duration=5)
        if not url:
             gr.Warning("请输入图片 URL。")
             yield (
                gr.State(), None, "请输入图片 URL。", gr.update(interactive=False), None, gr.update(interactive=True), gr.update(interactive=False)
             )
             return

        # 1. Disable generate, disable clear
        yield (
            gr.State(), # target_dir (no change yet)
            None,       # gallery
            gr.update(), # status
            gr.update(interactive=False), # reconstruct_btn
            url,        # url input
            gr.update(interactive=False), # generate_btn
            gr.update(interactive=False)  # clear_btn
        )
        
        # 2. Run generation
        session_dir, files, msg = generate_multiview_images(url, prompt_text if use_prompt else "")
        
        # 3. Handle result
        if "FAILED" in str(msg) or "ERROR" in str(msg) or "失败" in str(msg) or "无关" in str(msg) or "请输入" in str(msg):
             # Failure case
             if msg == "MEDICAL_ERROR":
                 gr.Warning("您上传的图片和医学无关")
                 msg = "就绪。"
                 url_out = ""
             elif msg == "Medical check failed (API Error).":
                 gr.Warning("医学检测失败（API 错误）请稍后重试。")
                 msg = "就绪。"
                 url_out = url
             elif msg == "请输入图片 URL。":
                 gr.Warning(msg)
                 msg = "就绪。"
                 url_out = url
             elif msg and str(msg).startswith("GENERATION_FAILED:"):
                 view_name = msg.split(":")[1]
                 gr.Warning(f"{view_name} 生成失败。请重试。")
                 msg = f"生成 {view_name} 失败。"
                 url_out = url
             elif msg and str(msg).startswith("GENERATION_ERROR:"):
                 parts = msg.split(":", 2)
                 view_name = parts[1]
                 error_detail = parts[2] if len(parts) > 2 else "未知错误"
                 gr.Warning(f"生成 {view_name} 出错: {error_detail}")
                 msg = f"生成 {view_name} 出错。"
                 url_out = url
             else:
                 url_out = url

             yield (
                None, 
                None, 
                msg, 
                gr.update(interactive=False), 
                url_out,
                gr.update(interactive=True),  # Re-enable generate
                gr.update(interactive=False)  # Keep clear disabled
             )
        else:
             # Success case
             yield (
                session_dir,
                files,
                msg,
                gr.update(interactive=True), # Enable reconstruct
                url,
                gr.update(interactive=False), # Keep generate DISABLED
                gr.update(interactive=True)   # Enable clear
             )

    def on_clear_url_click():
        return (
            None, # target_dir
            None, # gallery
            "已清空。准备生成。", # status
            gr.update(interactive=False), # reconstruct_btn
            "",   # url input
            gr.update(interactive=True),  # Enable generate
            gr.update(interactive=False)  # Disable clear
        )

    generate_btn.click(
        on_generate_click,
        inputs=[image_url_input, use_additional_prompt, additional_prompt_input],
        outputs=[target_dir_state, gallery, status_output, reconstruct_btn, image_url_input, generate_btn, clear_btn]
    )
    
    clear_btn.click(
        on_clear_click,
        outputs=[target_dir_state, gallery, status_output, reconstruct_btn, image_url_input, generate_btn, clear_btn]
    )

    generate_url_btn.click(
        on_generate_url_click,
        inputs=[manual_url_input, use_additional_prompt_url, additional_prompt_input_url],
        outputs=[target_dir_state, gallery, status_output, reconstruct_btn, manual_url_input, generate_url_btn, clear_url_btn]
    )
    
    clear_url_btn.click(
        on_clear_url_click,
        outputs=[target_dir_state, gallery, status_output, reconstruct_btn, manual_url_input, generate_url_btn, clear_url_btn]
    )
    
    upload_btn.click(
        handle_local_upload,
        inputs=[image_upload_input],
        outputs=[target_dir_state, gallery, status_output, reconstruct_btn]
    )
    
    def on_reconstruct(target_dir, conf, cam, sky, black, white, mode, 
                       process_res, infer_gs_val, ref_view, max_pts, gs_trj, gs_qual):
        # 1. Disable buttons immediately
        yield (
            gr.update(interactive=False, variant="secondary"), # reconstruct_btn
            "⏳ 开始重建...", # countdown_text
            None, # model_output
            gr.update(interactive=False), # generate_btn
            gr.update(interactive=False), # clear_btn
            gr.update(interactive=False), # upload_btn
            gr.update(visible=False), # download_model_btn
            gr.update(visible=False), # update_btn
            None, # gs_video
            gr.update(visible=not infer_gs_val), # gs_info
            gr.update(visible=infer_gs_val) # gs_video visibility
        )
        
        # 2. Run inference
        try:
            glb_path, gsvideo_path = run_depth_anything_inference(target_dir, conf, cam, sky, black, white, mode, 
                                                    process_res, infer_gs_val, ref_view, max_pts, gs_trj, gs_qual)
        except Exception as e:
            # Restore buttons on error. 
            yield (
                gr.update(interactive=True, variant="primary"), # reconstruct_btn
                f"❌ 错误: {e}", # countdown_text
                None, # model_output
                gr.update(interactive=True), # generate_btn
                gr.update(interactive=True), # clear_btn
                gr.update(interactive=True), # upload_btn
                gr.update(visible=False), # download_model_btn
                gr.update(visible=False), # update_btn
                None, # gs_video
                gr.update(visible=True), # gs_info
                gr.update(visible=False) # gs_video visibility
            )
            return

        # Determine display logic
        display_glb = glb_path
        status_suffix = ""
        show_update_btn = True
        if infer_gs_val:
            display_glb = None
            status_suffix = "\n\n(为显示更高质量的3d点云模型，请单独重建3d点云)"
            gr.Info("为显示更高质量的3d点云模型，请单独重建3d点云")
            show_update_btn = False

        # 3. Countdown loop
        for i in range(30, 0, -1):
            yield (
                gr.update(interactive=False, variant="secondary"), # reconstruct_btn
                f"⚠️ 冷却中: 请等待 {i} 秒...{status_suffix}", # countdown_text
                display_glb, # model_output
                gr.update(interactive=False), # generate_btn
                gr.update(interactive=False), # clear_btn
                gr.update(interactive=False), # upload_btn
                gr.update(value=glb_path, visible=True), # download_model_btn
                gr.update(visible=show_update_btn), # update_btn
                gsvideo_path, # gs_video
                gr.update(visible=not infer_gs_val), # gs_info
                gr.update(visible=infer_gs_val) # gs_video visibility
            )
            time.sleep(1)
            
        # 4. Re-enable. 
        yield (
            gr.update(interactive=True, variant="primary"), # reconstruct_btn
            f"✅ 准备好进行新的重建{status_suffix}", # countdown_text
            display_glb, # model_output
            gr.update(interactive=False), # generate_btn
            gr.update(interactive=True), # clear_btn
            gr.update(interactive=True), # upload_btn
            gr.update(value=glb_path, visible=True), # download_model_btn
            gr.update(visible=show_update_btn), # update_btn
            gsvideo_path, # gs_video
            gr.update(visible=not infer_gs_val), # gs_info
            gr.update(visible=infer_gs_val) # gs_video visibility
        )

    reconstruct_btn.click(
        on_reconstruct,
        inputs=[target_dir_state, conf_thres, show_cam, mask_sky, mask_black_bg, mask_white_bg, prediction_mode,
                process_res_method_dropdown, infer_gs, ref_view_strategy_dropdown, num_max_points, gs_trj_mode, gs_video_quality],
        outputs=[reconstruct_btn, countdown_text, model_output, generate_btn, clear_btn, upload_btn, download_model_btn, update_btn, gs_video, gs_info, gs_video]
    )
    
    # Update visualization when update button is clicked
    viz_inputs = [target_dir_state, conf_thres, show_cam, mask_sky, mask_black_bg, mask_white_bg, prediction_mode, num_max_points]
    update_btn.click(
        update_visualization,
        inputs=viz_inputs,
        outputs=[model_output, download_model_btn]
    )

if __name__ == "__main__":
    # share=True requires downloading frpc which might fail in some environments.
    # Setting share=False to avoid startup errors.
    # Port 6006 might be busy, using 6008
    demo.queue().launch(server_name="0.0.0.0", server_port=6008, share=False)

import os
import subprocess
import warnings
import sys
from typing import List, Dict
import logging
from datetime import datetime
import json
import ctypes
from pathlib import Path
import importlib.util
from importlib.metadata import version, PackageNotFoundError
from packaging import version as packaging_version
from control_vectors.create_control_vectors import main as create_control_vectors
from control_vectors.dataset_manager import DatasetManager
from control_vectors.hidden_state_data_manager import HiddenStateDataManager
from control_vectors.direction_analyzer import DirectionAnalyzer
from control_vectors.model_handler import ModelHandler

# Function to install missing packages
def install_packages():
    required = [
        'gradio==4.39.0',
        'huggingface_hub>=0.14.1',
        'repeng',
        'requests'  # Add requests for API calls
    ]

    for package in required:
        package_name = package.split('==')[0].split('>=')[0]
        spec = importlib.util.find_spec(package_name)

        if spec is None:
            print(f"Installing {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            if '==' in package or '>=' in package:
                try:
                    installed_version = version(package_name)
                    required_version = package.split('==')[-1] if '==' in package else package.split('>=')[-1]

                    if packaging_version.parse(installed_version) < packaging_version.parse(required_version):
                        print(f"Upgrading {package}")
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
                    else:
                        print(f"{package_name} is already up-to-date (version {installed_version})")
                except PackageNotFoundError:
                    print(f"Installing {package}")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            else:
                print(f"{package_name} is installed")

    try:
        import repeng
    except ImportError:
        print("Failed to import repeng. Please check the installation.")

# Install missing packages
install_packages()

import requests  # For API calls
import gradio as gr
import json
from huggingface_hub import hf_hub_download, snapshot_download
import numpy as np
import time
import random

# Function to install llama.cpp from source
def install_llama_cpp():
    llama_cpp_dir = os.path.join(os.getcwd(), "llama.cpp")
    if not os.path.exists(llama_cpp_dir):
        print("Installing llama.cpp from source")
        subprocess.check_call(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir])

        os.chdir(llama_cpp_dir)
        subprocess.check_call(["make"])
        os.chdir("..")
        print("llama.cpp installed successfully")
    else:
        print("llama.cpp installed")

# Install llama.cpp
install_llama_cpp()

# Function to start the llama server
def start_llama_server(model_file):
    llama_server_path = os.path.join(os.getcwd(), "llama.cpp", "llama-server")
    if not os.path.exists(llama_server_path):
        raise FileNotFoundError(
            f"Llama server executable not found at {llama_server_path}. Please check the installation.")

    command = [llama_server_path, "-m", model_file, "--host", "127.0.0.1", "--port", "8080"]

    try:
        # Start the server process and capture its output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Llama server process started.")

        # Wait a bit to allow the server to start
        time.sleep(5)

        # Check if the process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print("Server process terminated. Output:")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            raise RuntimeError("Llama server process terminated unexpectedly")

        return process
    except subprocess.CalledProcessError as e:
        print(f"Error starting llama server: {e}")
        raise

# Global variables
model = None
tokenizer = None
user_tag = None
asst_tag = None
saved_prompts_dict = {}
control_vector_options = []
current_model = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def log_info(message):
    logging.info(message)

def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    if 'model_filename' not in config:
        raise ValueError("model_filename is missing in config.json")
    return config


def is_model_downloaded(model_name, models_folder):
    model_path = Path(models_folder) / model_name
    return model_path.exists() and any(model_path.glob('*.bin'))


def download_model(model_name, model_filename, models_folder, hf_token):
    model_path = Path(models_folder) / model_name
    file_path = model_path / model_filename
    if file_path.exists():
        print(f"Model file {model_filename} is already downloaded.")
        return str(file_path)

    print(f"Downloading model file: {model_filename}")
    try:
        downloaded_path = hf_hub_download(
            repo_id=model_name,
            filename=model_filename,
            token=hf_token,
            cache_dir=model_path,
            resume_download=True,
        )
        print("Model file downloaded successfully")
        return downloaded_path
    except Exception as e:
        print(f"Error downloading model file: {e}")
        raise


def get_available_models():
    models_folder = Path('./models')
    print(f"Searching for models in: {models_folder}")
    if not models_folder.exists():
        print(f"Models folder not found: {models_folder}")
        return []

    models = []
    for root, dirs, files in os.walk(models_folder):
        for file in files:
            if file.endswith('.gguf'):
                relative_path = os.path.relpath(os.path.join(root, file), models_folder)
                models.append(relative_path)
                print(f"Found model: {relative_path}")

    if not models:
        print("No .gguf files found in the models folder or its subfolders.")
    else:
        print(f"Available models: {models}")

    return models

def refresh_model_list():
    models = get_available_models()
    return gr.Dropdown(choices=models, value=current_model)

def on_model_change(model_filename, progress=gr.Progress()):
    global current_model
    try:
        print(f"Attempting to change model to: {model_filename}")
        if model_filename != current_model:
            progress(0, desc="Starting model load")
            load_model(model_filename, progress)
            current_model = model_filename
            print(f"Model successfully changed to {model_filename}")
            progress(1, desc="Model load complete")
        else:
            print(f"Model {model_filename} is already loaded")
    except Exception as e:
        print(f"Error in on_model_change: {str(e)}")
        import traceback
        traceback.print_exc()
        progress(1, desc="Model load failed")
        raise gr.Error(f"Failed to load model: {str(e)}")

def load_model(model_filename=None):
    global user_tag, asst_tag, current_model

    try:
        print("Loading configuration")
        config = load_config()
        models_folder = config.get('models_folder', './models')
        hf_token = config.get('hf_token')

        if not hf_token:
            raise ValueError("Hugging Face token not found in config file")

        if model_filename is None:
            model_filename = config['model_filename']

        print("Locating model file")
        model_file = str(Path(models_folder) / model_filename)
        print(f"Attempting to load model: {model_file}")

        # Check if the model file exists, if not, download it
        if not os.path.exists(model_file):
            print(f"Model file not found: {model_file}")
            print("Downloading model file...")
            model_file = download_model(config['model_name'], model_filename, models_folder, hf_token)

        print(f"Model file found. Loading model: {model_file}")

        # If a model is already loaded, stop the current server
        if 'server_process' in globals():
            print("Stopping previous server")
            globals()['server_process'].terminate()
            globals()['server_process'].wait()

        print("Starting llama server")
        # Start the llama server
        server_process = start_llama_server(model_file)
        globals()['server_process'] = server_process

        print("Waiting for server to start")
        # Wait for the server to start and check its status
        server_url = "http://localhost:8080/v1/models"
        max_retries = 30
        retry_delay = 2

        for _ in range(max_retries):
            try:
                response = requests.get(server_url)
                if response.status_code == 200:
                    print("Llama server is running and ready")
                    break
            except requests.ConnectionError:
                print("Waiting for llama server to start...")
                if server_process.poll() is not None:
                    stdout, stderr = server_process.communicate()
                    print("Server process has terminated. Output:")
                    print("STDOUT:", stdout)
                    print("STDERR:", stderr)
                    raise RuntimeError("Llama server process terminated unexpectedly")
                time.sleep(retry_delay)
        else:
            raise RuntimeError("Failed to connect to llama server after multiple attempts")

        user_tag, asst_tag = "[INST]", "[/INST]"
        current_model = model_filename

        print(f"Model {model_filename} loaded successfully")
        return current_model  # Return the current model filename

    except Exception as e:
        print(f"Error in load_model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Initialize the model on startup
config = load_config()
current_model = config['model_filename']
initial_load_result = load_model()
print(initial_load_result)

# Base URL for the llama server
LLAMA_SERVER_URL = "http://localhost:8080/v1"

def api_call(endpoint, method="POST", data=None):
    """
    Make an API call to the llama server.

    :param endpoint: The API endpoint (e.g., "/chat/completions")
    :param method: HTTP method (default is "POST")
    :param data: Dictionary of data to send in the request body
    :return: JSON response from the server
    """
    url = f"{LLAMA_SERVER_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}

    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, data=json.dumps(data))
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"API call failed: {e}")
        return None


def llama_server_generate(prompt, max_tokens=512, temperature=0.7, top_p=0.95, top_k=50, repetition_penalty=1.0):
    """
    Generate text using the llama server's chat completions API.

    :param prompt: The input prompt for text generation
    :param max_tokens: Maximum number of tokens to generate
    :param temperature: Controls randomness in generation
    :param top_p: Nucleus sampling parameter
    :param top_k: Top-k sampling parameter
    :param repetition_penalty: Penalty for repeating tokens
    :return: Generated text or None if the API call fails
    """
    data = {
        "model": "default",  # Assuming a default model on the server
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }

    response = api_call("/chat/completions", data=data)

    if response and "choices" in response:
        return response["choices"][0]["message"]["content"]
    else:
        print("Failed to generate text")
        return None

def generate_text(prompt, max_new_tokens=512, temperature=0.9, top_p=0.95, top_k=50, repetition_penalty=1.0):
    global user_tag, asst_tag

    # Parse control vectors
    clean_prompt, vectors = parse_control_vectors(prompt)

    log_info(f"Chat Request. Prompt: {clean_prompt[:50]}...")  # Log first 50 chars of prompt

    full_prompt = f"{user_tag} {clean_prompt} {asst_tag}"

    try:
        if vectors:
            # TODO: Implement control vector logic for API-based approach
            log_info("Control vectors are not yet implemented for the API-based approach")

        # Generate text using the API
        response = llama_server_generate(full_prompt, max_tokens=max_new_tokens, temperature=temperature,
                                         top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty)

        if response:
            log_info(f"Generated response: {response[:50]}...")  # Log first 50 chars of response
            return response
        else:
            return "Failed to generate a response"

    except Exception as e:
        log_info(f"Error in generate_text: {e}")
        return f"An error occurred: {e}"

def apply_control_vector(control_model, control_vector, strength):
    try:
        control_model.set_control(control_vector, strength)
    except Exception as e:
        print(f"Error applying control vector: {e}")
        control_model.reset()  # Reset the model if there's an error


def chat(message, history, temperature, top_p, top_k, max_new_tokens, repetition_penalty):
    log_info(f"Received chat message: {message[:50]}...")  # Log first 50 chars of message
    # Define bos_token and eos_token
    bos_token = "<s>"
    eos_token = "</s>"

    # Construct the full prompt using the mistral chat template
    full_prompt = f"{bos_token}"
    for i, (user_msg, bot_msg) in enumerate(history):
        full_prompt += f"[INST] {user_msg} [/INST]"
        if bot_msg:
            full_prompt += f"{bot_msg}{eos_token}"
    
    full_prompt += f"[INST] {message} [/INST]"

    try:
        bot_message = generate_text(
            full_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )

        if bot_message:
            log_info(f"Generated bot message: {bot_message[:50]}...")  # Log first 50 chars of bot message
            new_history = history + [(message, bot_message)]
            log_info("Chat response generated successfully")
            return "", new_history
        else:
            log_info("Failed to generate chat response")
            bot_message = "I'm sorry, I couldn't generate a response. Please try again."
            new_history = history + [(message, bot_message)]
            return "", new_history
    except Exception as e:
        log_info(f"Error in chat function: {e}")
        bot_message = f"An error occurred: {e}"
        new_history = history + [(message, bot_message)]
        return "", new_history

def save_control_vector(vector, path):
    # Assuming the vector is a numpy array
    np.save(path, vector)
    log_info(f"Saved Control Vector to {path}")

def load_control_vector(path):
    try:
        vector = np.load(path)
        log_info(f"Loaded Control Vector from {path}")
        return vector
    except Exception as e:
        log_info(f"Error loading Control Vector from {path}: {e}")
        return None

def apply_control_vector(ctx, control_vector, strength):
    try:
        # Convert numpy array to ctypes array
        c_float_p = ctypes.POINTER(ctypes.c_float)
        vector_data = control_vector.ctypes.data_as(c_float_p)

        # Assuming llama.cpp has a function to apply control vectors
        success = llama.llama_apply_control_vector(ctx, vector_data, len(control_vector), strength)

        if not success:
            raise RuntimeError("Failed to apply control vector")

    except Exception as e:
        print(f"Error applying control vector: {e}")
        # Reset the model if there's an error
        llama.llama_reset(ctx)

def clear_chat():
    return []


def retry(history, temperature, top_p, top_k, max_new_tokens, repetition_penalty):
    if not history:
        return history

    # Find the last user message and its index
    last_user_message = None
    last_user_index = -1
    for i, message in enumerate(reversed(history)):
        if message[0] is not None:  # This is a user message
            last_user_message = message[0]
            last_user_index = len(history) - 1 - i
            break

    if last_user_message is None:
        log_info("No user message found for retry")
        return history

    # Keep the history up to and including the last user message
    new_history = history[:last_user_index + 1]

    # Prepare the full prompt
    full_prompt = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in new_history])
    full_prompt += f"\nHuman: {last_user_message}\nAI:"

    try:
        # Generate new response
        new_response = generate_text(
            full_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )

        if new_response:
            log_info(f"Generated new response for retry: {new_response[:50]}...")  # Log first 50 chars
            new_history.append((last_user_message, new_response))
            log_info("Retry successful")
        else:
            log_info("Failed to generate new response for retry")
            new_response = "I'm sorry, I couldn't generate a new response. Please try again."
            new_history.append((last_user_message, new_response))
    except Exception as e:
        log_info(f"Error in retry function: {e}")
        new_response = f"An error occurred during retry: {e}"
        new_history

def undo(history):
    if history:
        history.pop()
    return history

def load_prompts_from_file():
    global saved_prompts_dict
    saved_prompts_dict.clear()  # Clear existing prompts
    try:
        with open('savedprompts.txt', 'r') as f:
            for line in f:
                name, prompt = line.strip().split(':', 1)
                saved_prompts_dict[name] = prompt
    except FileNotFoundError:
        print("No saved prompts file found. Starting with an empty prompt list.")
    return list(saved_prompts_dict.keys())

def save_prompt_to_file(name, prompt):
    with open('savedprompts.txt', 'a') as f:
        f.write(f"{name}:{prompt}\n")

def save_prompt(prompt_name, prompt):
    if prompt_name and prompt:
        saved_prompts_dict[prompt_name] = prompt
        save_prompt_to_file(prompt_name, prompt)
        return gr.Dropdown(choices=list(saved_prompts_dict.keys()), value=prompt_name)
    return gr.Dropdown(choices=list(saved_prompts_dict.keys()))

def load_prompt(prompt_name):
    prompt = saved_prompts_dict.get(prompt_name, "")
    return {"value": prompt, "__type__": "update"}

# Load prompts at the start of the application
load_prompts_from_file()


def refresh_prompts():
    prompt_list = load_prompts_from_file()
    return gr.Dropdown(choices=prompt_list)


def load_control_vectors_from_folder():
    cv_folder = os.path.join(os.getcwd(), "cv")
    try:
        if not os.path.exists(cv_folder):
            os.makedirs(cv_folder)
            print(f"Created 'cv' folder at {cv_folder}")
        control_vector_options = ["None"] + [
            f for f in os.listdir(cv_folder)
            if f.endswith('.pt') and not f.endswith('_metadata.json')
        ]
        print(f"Control Vectors found: {control_vector_options}")
        return control_vector_options
    except Exception as e:
        print(f"Error loading Control Vectors: {e}")
        return ["None"]


def refresh_control_vectors():
    options = load_control_vectors_from_folder()
    print(f"Refreshed control vectors: {options}")
    return gr.Dropdown(choices=options, value="None", interactive=True, allow_custom_value=True)


control_vector_options = load_control_vectors_from_folder()


def add_control_vector_to_input(control_vector, vector_strength, current_input):
    if control_vector == "None":
        return current_input
    cv_name = control_vector.rsplit('.', 1)[0]  # Remove file extension if present
    cv_text = f"<{cv_name}:{vector_strength}>"
    log_info(f"Added Control Vector to input: {cv_text}")
    return current_input + " " + cv_text if current_input else cv_text


def parse_control_vectors(input_text):
    import re
    pattern = r'<(\w+):(-?\d+(?:\.\d+)?)>'
    matches = re.findall(pattern, input_text)
    vectors = [(name, float(strength)) for name, strength in matches]
    clean_text = re.sub(pattern, '', input_text).strip()
    return clean_text, vectors


def load_files_from_train_folder():
    train_folder = os.path.join(os.getcwd(), "train")
    try:
        # Create the train folder if it doesn't exist
        os.makedirs(train_folder, exist_ok=True)
        
        files = [f for f in os.listdir(train_folder) if os.path.isfile(os.path.join(train_folder, f))]
        return ["None"] + files  # Add "None" as the first option
    except Exception as e:
        print(f"Error loading files: {e}")
        return ["None"]  # Return ["None"] if there's an error


def refresh_file_list():
    files = load_files_from_train_folder()
    return {"choices": files, "__type__": "update"}, {"choices": files, "__type__": "update"}, {"choices": files, "__type__": "update"}


def load_file_content(file_name: str) -> List[str]:
    if file_name == "None":
        return []
    train_folder = os.path.join(os.getcwd(), "train")
    file_path = os.path.join(train_folder, file_name)
    with open(file_path, 'r') as f:
        return json.load(f)


def make_dataset(prompt_stems, continuations, positive_persona, negative_persona):
    global model, user_tag, asst_tag

    # Load prompt stems and continuations
    prompt_stems_suffixes = load_file_content(prompt_stems)
    continuations_suffixes = load_file_content(continuations)

    # Tokenize and truncate prompt stems
    truncated_prompt_stems = []
    for s in prompt_stems_suffixes:
        tokens = model.tokenize(s.encode('utf-8'))
        for i in range(1, len(tokens)):
            truncated_prompt_stems.append(model.detokenize(tokens[:i]).decode('utf-8'))

    # Tokenize and truncate continuations
    truncated_continuations = []
    for s in continuations_suffixes:
        tokens = model.tokenize(s.encode('utf-8'))
        for i in range(1, len(tokens) - 5):
            truncated_continuations.append(model.detokenize(tokens[:i]).decode('utf-8'))

    # Combine truncated prompt stems and continuations
    all_suffixes = truncated_prompt_stems + truncated_continuations

    # Create dataset entries
    dataset = []
    for suffix in all_suffixes:
        positive_entry = f"{user_tag} {positive_persona} {asst_tag} {suffix}"
        negative_entry = f"{user_tag} {negative_persona} {asst_tag} {suffix}"
        dataset.append({
            "positive": positive_entry,
            "negative": negative_entry
        })

    return {"dataset": dataset}

def save_dataset(dataset: Dict[str, List[Dict[str, str]]], dataset_name: str):
    dataset_folder = os.path.join(os.getcwd(), "dataset")
    os.makedirs(dataset_folder, exist_ok=True)
    file_path = os.path.join(dataset_folder, f"{dataset_name}.json")

    with open(file_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    return file_path


# Update the existing make_dataset function to use the new implementation
def train_vector(vector_name, dataset_file, default_strength, progress=gr.Progress()):
    global model

    if not vector_name:
        return "Please enter a Control Vector Name."

    if dataset_file == "None":
        return "Please select a dataset for training."

    # Load the dataset
    dataset_folder = os.path.join(os.getcwd(), "dataset")
    dataset_path = os.path.join(dataset_folder, dataset_file)

    try:
        with open(dataset_path, 'r') as f:
            dataset_json = json.load(f)
    except Exception as e:
        return f"Error loading dataset: {e}"

    # Convert the loaded JSON data to the format expected by llama.cpp
    dataset = [
        (entry["positive"], entry["negative"])
        for entry in dataset_json["dataset"]
    ]

    # Create a new context for training
    params = llama.llama_context_default_params()
    params.n_ctx = 2048
    params.n_threads = 4
    ctx = llama.llama_new_context_with_model(model, params)

    if not ctx:
        return "Failed to create new context for training"

    # Train the control vector
    try:
        progress(0, desc="Training Control Vector")

        # Assuming llama.cpp has a function to train control vectors
        control_vector = (ctypes.c_float * 4096)()  # Adjust size as needed
        success = llama.llama_train_control_vector(ctx, dataset, len(dataset), control_vector, 4096)

        if not success:
            raise RuntimeError("Control vector training failed")

        progress(1, desc="Training Complete")
    except Exception as e:
        llama.llama_free(ctx)
        return f"Error during training: {e}"

    # Save the trained vector
    cv_folder = os.path.join(os.getcwd(), "cv")
    os.makedirs(cv_folder, exist_ok=True)
    vector_path = os.path.join(cv_folder, f"{vector_name}.npy")

    try:
        save_control_vector(np.ctypeslib.as_array(control_vector), vector_path)
    except Exception as e:
        llama.llama_free(ctx)
        return f"Error saving control vector: {e}"

    # Save metadata
    metadata_path = os.path.join(cv_folder, f"{vector_name}_metadata.json")
    metadata = {
        "name": vector_name,
        "default_strength": default_strength,
        "dataset": dataset_file
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    llama.llama_free(ctx)
    return f"Vector '{vector_name}' trained successfully using dataset '{dataset_file}'. Saved to {vector_path}"

def generate_dataset_name(positive_persona: str, negative_persona: str) -> str:
    # Clean and truncate personas to create a valid filename
    def clean_name(name: str) -> str:
        return ''.join(c for c in name if c.isalnum() or c in [' ', '_']).rstrip()

    pos = clean_name(positive_persona)[:20]  # Limit to 20 characters
    neg = clean_name(negative_persona)[:20]
    base_name = f"{pos}_vs_{neg}_data"

    # Check if the name already exists and add version number if needed
    dataset_folder = os.path.join(os.getcwd(), "dataset")
    version = 1
    new_name = base_name
    while os.path.exists(os.path.join(dataset_folder, f"{new_name}.json")):
        version += 1
        new_name = f"{base_name}_v{version}"

    return new_name


def load_datasets_from_folder():
    dataset_folder = os.path.join(os.getcwd(), "dataset")
    try:
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
            print(f"Created 'dataset' folder at {dataset_folder}")

        files = ["None"] + [f for f in os.listdir(dataset_folder) if f.endswith('.json')]
        print(f"Datasets found: {files}")
        return {"choices": files, "__type__": "update"}
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return {"choices": ["None"], "__type__": "update"}


def train_vector(vector_name, dataset_file, default_strength, progress=gr.Progress()):
    if not vector_name:
        return "Please enter a Control Vector Name."

    if dataset_file == "None":
        return "Please select a dataset for training."

    # Load the dataset
    dataset_folder = os.path.join(os.getcwd(), "dataset")
    dataset_path = os.path.join(dataset_folder, dataset_file)

    try:
        with open(dataset_path, 'r') as f:
            dataset_json = json.load(f)
    except Exception as e:
        return f"Error loading dataset: {e}"

    # Prepare temporary files for create_control_vectors_main
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    prompt_stems_file = os.path.join(temp_dir, "prompt_stems.json")
    continuations_file = os.path.join(temp_dir, "continuations.json")
    writing_prompts_file = os.path.join(temp_dir, "writing_prompts.txt")

    # Prepare data for prompt_stems_file
    prompt_stems = {
        "pre": [""],  # Add appropriate pre-stems if needed
        "post": [""]  # Add appropriate post-stems if needed
    }
    with open(prompt_stems_file, 'w') as f:
        json.dump(prompt_stems, f)

    # Prepare data for continuations_file
    continuations = {
        "classes": ["positive", "negative"],
        "data": [
            [entry["positive"] for entry in dataset_json["dataset"]],
            [entry["negative"] for entry in dataset_json["dataset"]]
        ]
    }
    with open(continuations_file, 'w') as f:
        json.dump(continuations, f)

    # Prepare data for writing_prompts_file
    writing_prompts = [entry["positive"].split("]", 1)[-1].strip() for entry in dataset_json["dataset"]]
    with open(writing_prompts_file, 'w') as f:
        f.write("\n".join(writing_prompts))

    # Prepare output path
    cv_folder = os.path.join(os.getcwd(), "cv")
    os.makedirs(cv_folder, exist_ok=True)
    output_path = os.path.join(cv_folder, vector_name)

    try:
        progress(0, desc="Training Control Vector")
        create_control_vectors_main(
            model_id=current_model,
            output_path=output_path,
            prompt_stems_file_path=prompt_stems_file,
            continuations_file_path=continuations_file,
            writing_prompts_file_path=writing_prompts_file,
            num_prompt_samples=len(dataset_json["dataset"]),
            use_separate_system_message=False,
            skip_begin_layers=0,
            skip_end_layers=1,
            discriminant_ratio_tolerance=0.5
        )
        progress(1, desc="Training Complete")
    except Exception as e:
        return f"Error during training: {e}"

    # Clean up temporary files
    os.remove(prompt_stems_file)
    os.remove(continuations_file)
    os.remove(writing_prompts_file)

    # Save metadata
    metadata_path = os.path.join(cv_folder, f"{vector_name}_metadata.json")
    metadata = {
        "name": vector_name,
        "default_strength": default_strength,
        "dataset": dataset_file
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return f"Vector '{vector_name}' trained successfully using dataset '{dataset_file}'. Saved to {output_path}"

def save_control_vector(vector, path):
    with open(path, 'wb') as f:
        pickle.dump(vector, f)
    log_info(f"Saved Control Vector to {path}")

def load_control_vector(path):
    try:
        with open(path, 'rb') as f:
            vector = pickle.load(f)
        if not isinstance(vector, ControlVector):
            raise TypeError(f"Loaded object is not a ControlVector: {type(vector)}")
        log_info(f"Loaded Control Vector from {path}")
        return vector
    except Exception as e:
        log_info(f"Error loading Control Vector from {path}: {e}")
        return None

# Custom CSS
css = """
.message {
    padding: 10px;
    margin: 5px;
    border-radius: 15px;
}
.user {
    background-color: #8052e3;
}
.bot {
    background-color: #3d3d4a;
}
"""

# Create the Gradio interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Control Vector Interface")

    with gr.Row():
        with gr.Column(scale=4):
            available_models = get_available_models()
            model_dropdown = gr.Dropdown(
                choices=available_models,
                label="Select Model",
                value=current_model if current_model in available_models else None,
                interactive=True,
                allow_custom_value=True
            )
        with gr.Column(scale=1):
            refresh_models_button = gr.Button("⟳", size="sm")
            
    with gr.Tabs():
        with gr.TabItem("Chat"):
            chatbot = gr.Chatbot(
                label="Dialog",
                bubble_full_width=False,
                show_copy_button=True,
                height=480
            )
            msg = gr.Textbox(label="Type a message...", lines=1, interactive=True)

            with gr.Row():
                with gr.Column(scale=2):
                    submit = gr.Button("Submit", variant="primary")
                    clear_button = gr.Button("Clear")
                    retry_button = gr.Button("Retry")
                    undo_button = gr.Button("Undo")
                with gr.Column(scale=3):
                    prompts_dropdown = gr.Dropdown(choices=list(saved_prompts_dict.keys()), label="Prompt Select")
                    prompt_name = gr.Textbox(label="Prompt Name", lines=1)
                    with gr.Row():
                        save_prompt_button = gr.Button("Save Prompt")
                        refresh_button = gr.Button("⟳", size="sm")

            with gr.Row():
                with gr.Column(scale=3):
                    control_vector = gr.Dropdown(
                        choices=control_vector_options,
                        label="Control Vector",
                        value="None",
                        interactive=True,
                        allow_custom_value=True
                    )
                with gr.Column(scale=2):
                    vector_strength = gr.Slider(minimum=-5, maximum=5, value=0, step=0.1, label="Vector Strength")
                with gr.Column(scale=1, min_width=50):
                    add_cv_button = gr.Button("Add", size="sm")
                    cv_refresh_button = gr.Button("⟳", size="sm")

            with gr.Row():
                with gr.Column(scale=1):
                    temperature = gr.Slider(minimum=0, maximum=3.0, value=0.9, step=0.05, label="Temperature")
                    max_new_tokens = gr.Slider(minimum=0, maximum=8192, value=256, step=64, label="Max New Tokens")
                with gr.Column(scale=1):
                    top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="Top-p (nucleus sampling)")
                    top_k = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Top-k")
                with gr.Column(scale=1):
                    repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.2, step=0.05, label="Repetition penalty")

            model_dropdown.change(
                on_model_change,
                inputs=[model_dropdown],
                outputs=[],
                show_progress=True
            )

            refresh_models_button.click(
                refresh_model_list,
                outputs=[model_dropdown]
            )

            submit.click(chat, inputs=[msg, chatbot, temperature, top_p, top_k, max_new_tokens, repetition_penalty],
                         outputs=[msg, chatbot])
            msg.submit(chat, inputs=[msg, chatbot, temperature, top_p, top_k, max_new_tokens, repetition_penalty],
                       outputs=[msg, chatbot])
            retry_button.click(
                retry,
                inputs=[chatbot, temperature, top_p, top_k, max_new_tokens, repetition_penalty],
                outputs=[chatbot]
            )

            undo_button.click(undo, inputs=[chatbot], outputs=[chatbot])
            clear_button.click(clear_chat, outputs=[chatbot])

            save_prompt_button.click(
                save_prompt,
                inputs=[prompt_name, msg],
                outputs=[prompts_dropdown]
            )

            prompts_dropdown.change(
                load_prompt,
                inputs=[prompts_dropdown],
                outputs=[msg]
            )

            refresh_button.click(
                refresh_prompts,
                outputs=[prompts_dropdown]
            )
            cv_refresh_button.click(
                refresh_control_vectors,
                outputs=[control_vector]
            )

            add_cv_button.click(add_control_vector_to_input, inputs=[control_vector, vector_strength, msg],
                                outputs=[msg]
                                )

        with gr.TabItem("Train"):
            gr.Markdown("## Dataset Creation")
            with gr.Row():
                with gr.Column(scale=10):
                    prompt_stems = gr.Dropdown(choices=load_files_from_train_folder(), label="Prompt Stems", value="None", allow_custom_value=True)
                with gr.Column(scale=10):
                    continuations = gr.Dropdown(choices=load_files_from_train_folder(), label="Continuations", value="None", allow_custom_value=True)
                with gr.Column(scale=10):
                    creative_prompts = gr.Dropdown(choices=load_files_from_train_folder(), label="Creative Prompts", value="None", allow_custom_value=True)
                with gr.Column(scale=1, min_width=50):
                    refresh_button = gr.Button("⟳", size="sm")
            with gr.Row():
                positive_persona = gr.Textbox(label="Positive Persona")
                negative_persona = gr.Textbox(label="Negative Persona")
            with gr.Row():
                dataset_info = gr.Textbox(label="Dataset Info", interactive=False, scale=3)
                make_dataset_button = gr.Button("Make Dataset", scale=1)

            gr.Markdown("## Vector Training")
            with gr.Row():
                vector_name = gr.Textbox(label="Control Vector Name")
            with gr.Row():
                with gr.Column(scale=10):
                    dataset_dropdown = gr.Dropdown(
                        choices=load_datasets_from_folder()["choices"],
                        label="Select Dataset",
                        value="None",
                        allow_custom_value=True
                    )
                with gr.Column(scale=1, min_width=50):
                    dataset_refresh_button = gr.Button("⟳", size="sm")
            with gr.Row():
                default_strength = gr.Slider(minimum=-5, maximum=5, value=0, step=0.1, label="Default Vector Strength")
            with gr.Row():
                training_info = gr.Textbox(label="Training Info", interactive=False, scale=3)
                train_button = gr.Button("Train", scale=1, variant="primary")

            make_dataset_button.click(
                make_dataset,
                inputs=[prompt_stems, continuations, creative_prompts, positive_persona, negative_persona],
                outputs=[dataset_info]
            )

            refresh_button.click(
                refresh_file_list,
                outputs=[prompt_stems, continuations, creative_prompts]
            )

            train_button.click(
                train_vector,
                inputs=[vector_name, dataset_dropdown, default_strength],
                outputs=[training_info]
            )

            dataset_refresh_button.click(
                load_datasets_from_folder,
                outputs=[dataset_dropdown]
            )

demo.queue()
demo.launch(share=True)

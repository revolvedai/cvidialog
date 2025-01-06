import os
import subprocess
import warnings
import sys
from importlib.metadata import version, PackageNotFoundError
from typing import List, Dict
import logging
from datetime import datetime


# Function to install missing packages
def install_packages():
    from packaging.version import parse as parse_version
    from packaging.specifiers import SpecifierSet
    
    required = {
        'gradio': '>=4.39.0',
        'torch': '>=2.0.0',
        'transformers': '>=4.30.0',
        'huggingface_hub': '>=0.14.1',
        'tokenizers': '>=0.13.3',
        'protobuf': '>=3.20.0',
        'sentencepiece': '>=0.1.99',
        'accelerate': '>=0.26.0'  # Added accelerate
    }
    missing = []

    for package, version_constraint in required.items():
        try:
            installed_version = parse_version(version(package))
            spec = SpecifierSet(version_constraint)
            if installed_version not in spec:
                missing.append(f"{package}{version_constraint}")
        except PackageNotFoundError:
            missing.append(f"{package}{version_constraint}")

    if missing:
        print(f"Installing required packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *missing])
    else:
        print("All required packages are already installed with compatible versions.")

    # Install repeng from GitHub if needed
    try:
        import repeng
        print("repeng is already installed.")
    except ImportError:
        print("Installing repeng...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "git+https://github.com/vgel/repeng.git"])
        print("repeng installed successfully.")
# Install missing packages
install_packages()

import gradio as gr
import json
import torch

# Suppress flash attention warning if not compiled with it
warnings.filterwarnings("ignore", category=UserWarning, message=".*flash attention.*")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel, DatasetEntry
import pickle
from huggingface_hub import hf_hub_download, snapshot_download
import time
import torch.nn.functional as F
import random

# Global variables
model = None
tokenizer = None
user_tag = None
asst_tag = None
saved_prompts_dict = {}
control_vector_options = []

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def log_info(message):
    logging.info(message)

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)


def is_model_downloaded(model_name):
    cache_dir = os.path.join(os.getcwd(), "model_cache", model_name)
    return os.path.exists(cache_dir) and os.path.isdir(cache_dir) and os.listdir(cache_dir)


def download_model(model_name, hf_token):
    if is_model_downloaded(model_name):
        print(f"Model {model_name} is already downloaded.")
        return
    print(f"Downloading model: {model_name}")
    cache_dir = os.path.join(os.getcwd(), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        snapshot_download(
            repo_id=model_name,
            token=hf_token,
            cache_dir=cache_dir,
            local_dir=cache_dir
        )
        print("Model downloaded successfully")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise


def load_model(model_name=None):
    global model, tokenizer, user_tag, asst_tag

    config = load_config()
    model_name = model_name or config['model_name']
    hf_token = config['hf_token']

    print(f"Preparing to load model: {model_name}")

    # Ensure the model is downloaded
    download_model(model_name, hf_token)

    cache_dir = os.path.join(os.getcwd(), "model_cache")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir)
    tokenizer.pad_token_id = 0
    print("Tokenizer loaded successfully")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        token=hf_token,
        cache_dir=cache_dir,
        device_map="auto"
    )
    print("Base model loaded successfully")

    # Wrap with ControlModel
    print("Wrapping with ControlModel...")
    model = ControlModel(model, list(range(-5, -18, -1)))
    print("Model wrapped successfully")

    user_tag, asst_tag = "[INST]", "[/INST]"

    print("Model loading complete")
    return "Model loaded successfully"


config = load_config()
load_model(config['model_name'])


def generate_text(prompt, max_new_tokens=512, temperature=0.9, top_p=0.95, top_k=50, repetition_penalty=1.0):
    global model, tokenizer, user_tag, asst_tag

    # Parse control vectors
    clean_prompt, vectors = parse_control_vectors(prompt)
    
    # Log chat request
    log_info(f"Chat request received: {clean_prompt[:50]}...")
    
    # Create input prompt with instruction tags
    input_prompt = f"{user_tag} {clean_prompt} {asst_tag}"
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt").to(model.device)

    try:
        if vectors:
            outputs = []
            for vector_name, strength in vectors:
                vector_path = os.path.join(os.getcwd(), "cv", f"{vector_name}.pt")
                if not os.path.exists(vector_path):
                    log_info(f"Control Vector {vector_name} not found")
                    continue

                log_info(f"Applying Control Vector '{vector_name}' with strength {strength}")
                vector = load_control_vector(vector_path)
                if vector is None:
                    continue

                model.set_control(vector, strength)

                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                outputs.append(output)
                model.reset()

            combined_output = torch.cat(outputs, dim=0)
            generated_text = tokenizer.decode(combined_output[0], skip_special_tokens=False)
        else:
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=False)

        # Extract only the assistant's response after the last [/INST] tag
        response = generated_text.split(asst_tag)[-1].strip()
        
        # Double-check no instruction tags remain
        response = response.replace(user_tag, "").replace(asst_tag, "").strip()
        
        # Log the response
        log_info(f"Generated response: {response[:50]}...")
        
        return response

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
    global model, tokenizer

    if model is None or tokenizer is None:
        log_info("Model not loaded. Please load a model first.")
        return "", history + [("Model not loaded. Please load a model first.", None)]

    # Build conversation history with proper instruction tags
    conversation = ""
    for past_msg, past_resp in history:
        if past_msg is not None:  # Only process valid message pairs
            conversation += f"{user_tag} {past_msg} {asst_tag} {past_resp}\n"
    
    # Add current message without response
    full_prompt = f"{conversation}{message}"

    try:
        bot_message = generate_text(
            full_prompt, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        
        new_history = history + [(message, bot_message)]
        return "", new_history
    except Exception as e:
        log_info(f"Error in chat function: {e}")
        new_history = history + [(message, f"An error occurred: {e}")]
        return "", new_history
def apply_control_vector(control_model, control_vector, strength):
    try:
        control_model.set_control(control_vector, strength)
    except Exception as e:
        print(f"Error applying control vector: {e}")
        control_model.reset()  # Reset the model if there's an error

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
        return history

    # Keep the history up to and including the last user message
    new_history = history[:last_user_index + 1]

    # Prepare the full prompt using instruction format
    full_prompt = "\n".join([f"{user_tag} {h[0]} {asst_tag} {h[1]}" for h in new_history])
    full_prompt += f"\n{user_tag} {last_user_message} {asst_tag}"

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
        # Append the new response to the history
        new_history.append((last_user_message, new_response))
    except Exception as e:
        new_response = f"An error occurred: {e}"
        new_history.append((last_user_message, new_response))

    return new_history

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
        files = [f for f in os.listdir(train_folder) if os.path.isfile(os.path.join(train_folder, f))]
        return ["None"] + files  # Add "None" as the first option
    except Exception as e:
        print(f"Error loading files: {e}")
        return ["None"]  # Return ["None"] if there's an error


def refresh_file_list():
    files = load_files_from_train_folder()
    return {"choices": files, "__type__": "update"}, {"choices": files, "__type__": "update"}


def load_file_content(file_name: str) -> List[str]:
    if file_name == "None":
        return []
    train_folder = os.path.join(os.getcwd(), "train")
    file_path = os.path.join(train_folder, file_name)
    with open(file_path, 'r') as f:
        return json.load(f)


def make_dataset(
        truncated_outputs: str,
        true_facts: str,
        positive_persona: str,
        negative_persona: str,
        user_tag: str,
        asst_tag: str,
        tokenizer: AutoTokenizer
) -> Dict[str, List[Dict[str, str]]]:
    # Load truncated outputs and true facts
    output_suffixes = load_file_content(truncated_outputs)
    fact_suffixes = load_file_content(true_facts)

    # Tokenize and truncate outputs
    truncated_output_suffixes = [
        tokenizer.convert_tokens_to_string(tokens[:i])
        for tokens in (tokenizer.tokenize(s) for s in output_suffixes)
        for i in range(1, len(tokens))
    ]

    # Tokenize and truncate facts
    truncated_fact_suffixes = [
        tokenizer.convert_tokens_to_string(tokens[:i])
        for tokens in (tokenizer.tokenize(s) for s in fact_suffixes)
        for i in range(1, len(tokens) - 5)
    ]

    # Combine truncated outputs and facts
    all_suffixes = truncated_output_suffixes + truncated_fact_suffixes

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
def make_dataset_ui(truncated_outputs, true_facts, positive_persona, negative_persona):
    global tokenizer, user_tag, asst_tag

    if not all([truncated_outputs, true_facts, positive_persona, negative_persona]):
        return "Please fill in all fields."

    if truncated_outputs == "None" or true_facts == "None":
        return "Please select valid files for Truncated Outputs and True Facts."

    dataset = make_dataset(
        truncated_outputs,
        true_facts,
        positive_persona,
        negative_persona,
        user_tag,
        asst_tag,
        tokenizer
    )

    # Generate a unique name for the dataset using the new function
    dataset_name = generate_dataset_name(positive_persona, negative_persona)

    file_path = save_dataset(dataset, dataset_name)
    response_count = len(dataset["dataset"])

    return f"Dataset '{dataset_name}' created successfully with {response_count} positive/negative pairs. Saved to {file_path}"


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
    global model, tokenizer

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

    # Convert the loaded JSON data to DatasetEntry objects
    dataset = [DatasetEntry(positive=entry["positive"], negative=entry["negative"])
               for entry in dataset_json["dataset"]]

    # Reset the model before training
    model.reset()

    # Train the control vector
    try:
        progress(0, desc="Training Control Vector")
        control_vector = ControlVector.train(model, tokenizer, dataset)
        progress(1, desc="Training Complete")
    except Exception as e:
        return f"Error during training: {e}"

        # Save the trained vector
    cv_folder = os.path.join(os.getcwd(), "cv")
    os.makedirs(cv_folder, exist_ok=True)
    vector_path = os.path.join(cv_folder, f"{vector_name}.pt")

    try:
        save_control_vector(control_vector, vector_path)
    except Exception as e:
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

    return f"Vector '{vector_name}' trained successfully using dataset '{dataset_file}'. Saved to {vector_path}"

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

# Available models and other options
model_options = ["mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-2-7b-chat-hf", "ggml model (upload .bin file)"]

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
            model_dropdown = gr.Dropdown(choices=model_options, label="Select Model", value=model_options[0])

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
                    truncated_outputs = gr.Dropdown(choices=load_files_from_train_folder(), label="Truncated Outputs", value="None", allow_custom_value=True)
                with gr.Column(scale=10):
                    true_facts = gr.Dropdown(choices=load_files_from_train_folder(), label="True Facts", value="None", allow_custom_value=True)
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
                make_dataset_ui,
                inputs=[truncated_outputs, true_facts, positive_persona, negative_persona],
                outputs=[dataset_info]
            )

            refresh_button.click(
                refresh_file_list,
                outputs=[truncated_outputs, true_facts]
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
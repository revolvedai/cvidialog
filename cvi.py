import os
import subprocess
import warnings
import sys
from importlib.metadata import version, PackageNotFoundError

# Function to install missing packages
def install_packages():
    required = ['gradio==3.0.0', 'torch==2.0.1', 'transformers==4.28.1', 'huggingface_hub==0.13.4']
    missing = []

    for package in required:
        try:
            pkg_name = package.split('==')[0]
            version(pkg_name)
        except PackageNotFoundError:
            missing.append(package)

    if missing:
        print(f"Missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

    # Install repeng from GitHub
    try:
        import repeng
        print("repeng is already installed.")
    except ImportError:
        print("repeng not found. Installing...")
        subprocess.check_call(["pip", "install", "git+https://github.com/vgel/repeng.git"])
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
from huggingface_hub import hf_hub_download, snapshot_download
import time
import torch.nn.functional as F
import random

# Global variables
model = None
tokenizer = None
user_tag = None
asst_tag = None

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def download_model(model_name, hf_token):
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

    # First, ensure the model is downloaded
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

# Load the model at startup
load_model()

def generate_text(prompt, max_new_tokens=512, temperature=0.9, top_p=0.95, top_k=50, repetition_penalty=1.0):
    global model, tokenizer, user_tag, asst_tag

    # Input validation
    temperature = max(0.01, min(3.0, temperature))
    if abs(temperature - 2.0) < 1e-6:
        temperature += 1e-6
    top_p = max(0.0, min(1.0, top_p))
    top_k = max(1, int(top_k))
    repetition_penalty = max(1.0, repetition_penalty)
    max_new_tokens = min(int(max_new_tokens), 8192)

    input_ids = tokenizer.encode(f"{user_tag} {prompt} {asst_tag}", return_tensors="pt").to(model.device)

    try:
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

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        parts = generated_text.split(asst_tag)
        if len(parts) > 1:
            return parts[-1].strip()
        else:
            return generated_text.strip()

    except Exception as e:
        print(f"Error in generate_text: {e}")
        return f"An error occurred: {e}"

def chat(message, history, temperature, top_p, top_k, max_new_tokens, repetition_penalty):
    global model, tokenizer

    if model is None or tokenizer is None:
        return "", history + [("Model not loaded. Please load a model first.", None)]

    full_prompt = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in history])
    full_prompt += f"\nHuman: {message}\nAI:"

    try:
        bot_message = generate_text(full_prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                                    top_k=top_k, repetition_penalty=repetition_penalty)
        history.append((message, bot_message))
    except Exception as e:
        bot_message = f"An error occurred: {e}"
        history.append((message, bot_message))

    return "", history
def clear_chat():
    return []

# Placeholder functions (to be implemented later)

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

def save_prompt(prompt_name, prompt):
    return f"Prompt '{prompt_name}' saved: {prompt}"


def make_dataset(prefix_list, suffix_list, positive_persona, negative_persona):
    return f"Dataset created with prefix: {prefix_list}, suffix: {suffix_list}, positive: {positive_persona}, negative: {negative_persona}"


def train_vector(vector_name, default_strength, dataset_info, progress=gr.Progress()):
    for i in range(100):
        progress(i / 100, desc="Training")
    return f"Vector '{vector_name}' trained with default strength {default_strength}. Dataset: {dataset_info}"


# Available models and other options
model_options = ["mistralai/Mistral-7B-Instruct-v0.1", "meta-llama/Llama-2-7b-chat-hf", "ggml model (upload .bin file)"]
control_vector_options = ["None", "Creative", "Formal", "Casual"]
prefix_options = ["Prefix List 1", "Prefix List 2", "Prefix List 3"]
suffix_options = ["Suffix List 1", "Suffix List 2", "Suffix List 3"]
saved_prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

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
with gr.Blocks(css=css) as demo:  # Remove js=js_code from here
    gr.Markdown("# Control Vector Interface")

    with gr.Row():
        with gr.Column(scale=4):
            model_dropdown = gr.Dropdown(choices=model_options, label="Select Model", value=model_options[0])

    with gr.Tabs():
        with gr.TabItem("Chat"):
            chatbot = gr.Chatbot(label="Dialog", bubble_full_width=False, show_copy_button=True)
            msg = gr.Textbox(label="Type a message...", lines=1)

            with gr.Row():
                with gr.Column(scale=2):
                    submit = gr.Button("Submit", variant="primary")
                    clear_button = gr.Button("Clear")
                    retry_button = gr.Button("Retry")
                    undo_button = gr.Button("Undo")
                with gr.Column(scale=3):
                    prompt_name = gr.Textbox(label="Prompt Name", lines=1)
                    save_prompt_button = gr.Button("Save Prompt")
                    prompts_dropdown = gr.Dropdown(choices=saved_prompts, label="Prompt Select")

            with gr.Row():
                control_vector = gr.Dropdown(choices=control_vector_options, label="Control Vector", value="None", scale=3)
                vector_strength = gr.Slider(minimum=-5, maximum=5, value=0, step=0.1, label="Vector Strength", scale=2)
                add_button = gr.Button("Add", size="sm")

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
            save_prompt_button.click(save_prompt, inputs=[prompt_name, msg], outputs=[prompts_dropdown])

        with gr.TabItem("Train"):
            gr.Markdown("## Dataset Creation")
            with gr.Row():
                prefix_list = gr.Dropdown(choices=prefix_options, label="Prefix List")
                suffix_list = gr.Dropdown(choices=suffix_options, label="Suffix List")
            with gr.Row():
                positive_persona = gr.Textbox(label="Positive Persona")
                negative_persona = gr.Textbox(label="Negative Persona")
            with gr.Row():
                dataset_info = gr.Textbox(label="Dataset Info", interactive=False, scale=3)
                make_dataset_button = gr.Button("Make Dataset", scale=1)

            gr.Markdown("## Vector Training")
            with gr.Row():
                vector_name = gr.Textbox(label="Control Vector Name")
                default_strength = gr.Slider(minimum=-5, maximum=5, value=0, step=0.1, label="Default Vector Strength")
            with gr.Row():
                training_info = gr.Textbox(label="Training Info", interactive=False, scale=3)
                train_button = gr.Button("Train", scale=1, variant="primary")

            make_dataset_button.click(
                make_dataset,
                inputs=[prefix_list, suffix_list, positive_persona, negative_persona],
                outputs=[dataset_info]
            )

            train_button.click(
                train_vector,
                inputs=[vector_name, default_strength, dataset_info],
                outputs=[training_info]
            )

demo.queue()
demo.launch()

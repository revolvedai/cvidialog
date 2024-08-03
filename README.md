# Control Vector Interface (CVI)

CVI is a Gradio-based web interface for interacting with and training Control Vectors, a powerful technique for steering large language models (LLMs) without fine-tuning. This project leverages the [repeng](https://github.com/vgel/repeng) library created by Theia Vogel to provide an intuitive UI for chatting with LLMs, applying Control Vectors, and training new vectors.

## Features

- Chat with LLMs using a familiar chat interface
- Apply pre-trained Control Vectors to steer model outputs
- Train new Control Vectors using custom datasets
- Manage prompts and datasets
- Support for multiple LLM models

## Installation

1. Clone this repository:
   ```
   git clone [https://github.com/revolvedai/cvidialog.git](https://github.com/revolvedai/cvidialog.git)
   cd cvidialog
   ```


2. Modify config.json to use your Huggingface Token
   ```
   hf_token: yourhuggingfacetokenhere
   ```
   
3. Run the application and it will install the required packages as well as download Mistral Instruct 7B 0.1

## Usage

### Chatting

1. Type your message in the text box and click "Submit" or press Enter
2. Adjust generation parameters like temperature, top-p, top-k, repetition penalty and max new tokens as needed
3. Clear, Retry, Undo to clean up the chat dialog.

### Using Control Vectors

1. Select a Control Vector from the dropdown menu
2. Adjust the vector strength using the slider
3. Click "Add" to apply the vector to your next message.
4. This will place the controlvector command into your chat box, where the strength can be modified. This is stripped out of the prompt that is passed to the agent.

### Training Control Vectors

1. Go to the "Train" tab
2. Create a dataset by selecting truncated outputs and true facts files, and specifying positive and negative personas
3. Click "Make Dataset" to generate a new dataset
4. Enter a name for your new Control Vector
5. Refresh the dataset folder dropdown
6. Select the dataset you just created
7. Click "Train" to train a new Control Vector

## Acknowledgements

This project is built on top of the [repeng](https://github.com/vgel/repeng) library created by Theia Vogel. We are grateful for their work in making Control Vectors accessible and easy to use.

---

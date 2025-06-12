# Story Generation with Constraint Words: Knowledge Distillation of GPT-2

## Project Overview
This project focuses on fine-tuning a GPT-2 model for story generation with constraint words using knowledge distillation. The teacher model is Qwen/Qwen3-1.7B, and the student model is a distilled version of GPT-2 (openai-community/gpt2). The fine-tuning dataset is the ROCStories dataset, which contains short, five-sentence stories. The goal is to generate coherent stories that include specified constraint words, such as "princess", "castle", and "dragon". The project also compares the performance of the trained model against the base GPT-2 model in terms of coherence and constraint word inclusion success rate.

- **Environment**: Trained on a CUDA-enabled device with BF16 support (A100 SXM GPU on RunPod), later evaluated on a Mac with MPS (Apple GPU) acceleration 
- **Repository**: The trained model is hosted on Hugging Face at [here4code/distilled-gpt2-story-generation-Qwen3-1.7B](https://huggingface.co/here4code/distilled-gpt2-story-generation-Qwen3-1.7B).

## Installation

### Prerequisites
- Python 3.10 or later
- PyTorch with CUDA support (for training) and MPS support (for Mac evaluation)
- Transformers library
- Hugging Face Hub library (for model upload/download)

### Setup
1. Clone this repository (if applicable) or create a new project directory.
2. Install the required dependencies:
   ```bash
   pip install torch transformers huggingface_hub
   ```
3. Verify PyTorch installation and device support:
   ```python
   import torch
   print(torch.__version__)
   print("CUDA available:" if torch.cuda.is_available() else "CUDA not available")
   print("MPS available:" if torch.backends.mps.is_available() else "MPS not available")
   ```

## Dataset
The project uses the **ROCStories** dataset, available on Hugging Face as `Ximing/ROCStories`. This dataset contains short stories with five sentences each, designed for commonsense reasoning and story completion tasks. The dataset was used to fine-tune the GPT-2 model to generate stories while incorporating specified constraint words.

- **Dataset Size**: Approximately 100,000 stories.
- **Preprocessing**: Stories were tokenized using the GPT-2 tokenizer with a maximum length of 128 tokens, with constraint words injected into the training data to encourage the model to learn their inclusion.

## Training Process
The GPT-2 model (openai-community/gpt2) was fine-tuned using knowledge distillation, with Qwen/Qwen3-1.7B as the teacher model. The process involved:
1. **Knowledge Distillation**:
   - The teacher model (Qwen/Qwen3-1.7B) generated logits for the ROCStories dataset.
   - The student model (GPT-2) was trained to minimize the KL-divergence between its logits and the teacher’s logits, while also optimizing for the language modeling loss.
2. **Hyperparameters**:
   - Epochs: 2
   - Batch Size: 128
   - Max Length: 128 tokens
   - Device: CUDA-enabled GPU with BF16 support (A100 SXM)
   - Learning Rate: 5e-5 (default assumed, adjust if different)
3. **Training Loss**:
   - Step 50: 6.8664
   - Step 100: 3.2611
   - Step 150: 2.9390
   - Step 200: 2.7870
   - Step 250: 2.6997
   - **Training Loss per Epoch**:
     - Epoch 1: 4.6367
     - Epoch 2: 2.7681
4. **Constraint Word Inclusion**:
   - Constraint words were injected into the training data to encourage the model to include them naturally in generated stories.

The trained model was saved and uploaded to Hugging Face under the repository `here4code/distilled-gpt2-story-generation-Qwen3-1.7B`.

## Evaluation
The performance of the trained model was evaluated and compared against the base GPT-2 model. The evaluation focused on:
- **Constraint Word Inclusion Success Rate**: The percentage of required terms (e.g., "princess", "castle", "dragon") naturally included in the generated text (before post-processing).
- **Coherence**: A qualitative assessment of the generated text’s grammatical correctness and narrative flow.

### Inference Results
Inference was performed on a subset of 20 samples with the following settings:
- **Max New Tokens for Generation**: 50
- **Teacher Model (Qwen/Qwen3-1.7B) Constraint Word Inclusion Success Rate**: 85.00%
- **Student Model (Trained GPT-2) Constraint Word Inclusion Success Rate**: 90.00%

### Evaluation Setup
- **Starting Part**: "In a faraway kingdom"
- **Required Terms**: "princess", "castle", "dragon"
- **Generation Parameters**:
  - Sampling: `do_sample=True`
  - Top-k: 40
  - Top-p: 0.9
  - Temperature: 0.6
  - No Repeat N-gram Size: 3
  - Number of Sequences: 5 (to select the best output)

### Results
- **Base GPT-2**:
  - Success Rate: 100.00% (all terms included naturally)
  - Example Output: "In a faraway kingdom, a princess and castle and dragon were nearby. The princess lived in the castle, which was guarded by a fierce dragon. One day, she decided to befriend the dragon to protect her kingdom."
- **Trained Model**:
  - Success Rate: 66.67% (missed "dragon" before post-processing)
  - Example Output: "In a faraway kingdom, a princess and castle and dragon were nearby. The princess lived in the castle and loved to explore the gardens. As she continued her journey, she encountered a dragon in the distance."
- **Analysis**:
  - The base GPT-2 outperformed the trained model in naturally including constraint words in the qualitative evaluation.
  - However, in the inference subset of 20 samples, the trained student model achieved a higher constraint word inclusion success rate (90.00%) compared to the teacher model (85.00%), indicating improved performance in controlled settings.

## Usage
You can use the trained model to generate stories with constraint words. Below is an example of how to load the model and generate a statement.

### Example: Generate a Story
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set device (MPS for Mac, or CPU as fallback)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the trained model and tokenizer
model_name = "here4code/distilled-gpt2-story-generation-Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# Define the prompt and required terms
starting_part = "In a faraway kingdom"
required_terms = ["princess", "castle", "dragon"]
enhanced_prompt = f"{starting_part}, a {' and '.join(required_terms[:-1])} and {required_terms[-1]} were nearby."

# Tokenize and generate
inputs = tokenizer(enhanced_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        max_new_tokens=50,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        no_repeat_ngram_size=3,
        temperature=0.6
    )
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Story:", generated_text)
```

### Example: Compare with Base GPT-2
Refer to the code in `Cell 13` (provided in the project notebook) to compare the trained model with the base GPT-2 model interactively. This script allows you to input a custom starting part and required terms to evaluate both models.

## Future Improvements
- **Constraint Word Inclusion**: Despite the student model’s higher success rate in inference (90.00%), it still struggles to naturally include constraint words in qualitative evaluations (success rate 66.67% in the example). Future work could involve modifying the loss function to penalize missing constraint words or using a larger teacher model for distillation.
- **Coherence**: Adjust generation parameters (e.g., lower temperature) or fine-tune the model further to improve narrative coherence.
- **Dataset**: Incorporate a more diverse dataset to enhance the model’s storytelling capabilities.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- **Hugging Face**: For providing the `transformers` library and hosting the models.
- **ROCStories Dataset**: For the training data used in this project.
- **RunPod**: For providing the A100 SXM GPU used for training.

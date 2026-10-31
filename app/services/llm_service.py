import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 4-bit quantization config taaki GTX 1650 (4GB VRAM) par model fit ho jaye
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model_name = "microsoft/Phi-3-mini-4k-instruct"

print(f"Loading local LLM ({model_name}) into GTX 1650 GPU...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map={"": 0}
)

def generate_rag_answer(context: str, question: str) -> str:
    """
    Retrieved context aur user question ko Phi-3 prompt format mein dal kar 
    local GPU par answer generate karega.
    """
    prompt = f"""<|system|>
You are a secure religious AI assistant. Answer the user's question accurately and spritually using ONLY the provided context. 
If the answer cannot be found in the context, state "I cannot find the answer in the provided documents."<|end|>
<|user|>
Context:
{context}

Question:
{question}<|end|>
<|assistant|>
"""
    
    # Tokenize input and move to GPU
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate response locally via PyTorch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2, # Low temperature for factual enterprise answers
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Input prompt ki length ke hisab se slice karna sabse safe tarika hai
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return answer

from transformers import AutoProcessor, AutoModelForSeq2SeqLM

# Load model
print("Loading T5Gemma-2...")
processor = AutoProcessor.from_pretrained("google/t5gemma-2-270m-270m")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5gemma-2-270m-270m")

def few_shot_translate(text_to_translate: str, num_examples: int = 3):
    """
    Translate using few-shot learning with examples
    """
    
    # English-Persian translation examples (high quality pairs)
    translation_examples = [
        ("Hello", "سلام"),
        ("Good morning", "صبح بخیر"),
        ("How are you?", "تو خوبی؟"),
        ("What is your name?", "نام تو چیه؟"),
        ("My name is Ali", "نام من علی است"),
        ("Thank you", "متشکرم"),
        ("You're welcome", "خوش آمدی"),
        ("Goodbye", "خداحافظ"),
        ("Yes", "بله"),
        ("No", "خیر"),
        ("Please", "لطفا"),
        ("Excuse me", "ببخشید"),
        ("I don't understand", "من متوجه نمی‌شوم"),
        ("Do you speak English?", "تو انگلیسی حرف می‌زنی؟"),
        ("Where is the bathroom?", "دستشویی کجاست؟"),
    ]
    
    # Build few-shot prompt with examples
    prompt = "Translate English to Persian:\n\n"
    
    # Add examples (use first num_examples)
    for english, persian in translation_examples[:num_examples]:
        prompt += f"English: {english}\nPersian: {persian}\n\n"
    
    # Add the text to translate
    prompt += f"English: {text_to_translate}\nPersian:"
    
    print(f"\n{'='*70}")
    print(f"PROMPT:\n{prompt}")
    print(f"{'='*70}\n")
    
    # Generate translation
    inputs = processor(text=prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.1,      # Low temperature = more deterministic
        do_sample=False,      # Greedy decoding
        top_p=0.9,
    )
    
    # Decode and extract translation
    full_output = processor.decode(outputs[0], skip_special_tokens=True)
    
    # The output will have the full prompt + translation
    # Extract just the translation part
    if "Persian:" in full_output:
        translation = full_output.split("Persian:")[-1].strip()
    else:
        translation = full_output
    
    return translation.strip()

# Test 1: Simple translation
print("TEST 1: Simple greeting")
print("="*70)
result = few_shot_translate("Hi, how are you?", num_examples=3)
print(f"Input: 'Hi, how are you?'")
print(f"Output: {result}\n")

# Test 2: Longer sentence
print("\nTEST 2: Longer sentence")
print("="*70)
result = few_shot_translate("What is your name?", num_examples=5)
print(f"Input: 'What is your name?'")
print(f"Output: {result}\n")

# Test 3: Multiple sentences
print("\nTEST 3: Multiple sentences")
print("="*70)
result = few_shot_translate("Hello. My name is John. How are you?", num_examples=4)
print(f"Input: 'Hello. My name is John. How are you?'")
print(f"Output: {result}\n")

# Test 4: Question
print("\nTEST 4: More complex question")
print("="*70)
result = few_shot_translate("Do you speak English?", num_examples=6)
print(f"Input: 'Do you speak English?'")
print(f"Output: {result}\n")

# Test 5: Your original request
print("\nTEST 5: Original request")
print("="*70)
result = few_shot_translate("Hi What is your name?", num_examples=5)
print(f"Input: 'Hi What is your name?'")
print(f"Output: {result}\n")

# Test 6: With more examples (better accuracy)
print("\nTEST 6: With more examples (10 examples)")
print("="*70)
result = few_shot_translate("Excuse me, where is the bathroom?", num_examples=10)
print(f"Input: 'Excuse me, where is the bathroom?'")
print(f"Output: {result}\n")
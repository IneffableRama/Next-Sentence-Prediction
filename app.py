from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util
import language_tool_python
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
tool = language_tool_python.LanguageTool('en-US')

def generate_next_sentence(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=len(inputs[0]) + max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    next_part = generated_text[len(prompt):].strip()
    return next_part

def grammar_check(sentence):
    matches = tool.check(sentence)
    return len(matches) == 0, matches

def relevance_score(input_text, generated_text):
    embeddings = semantic_model.encode([input_text, generated_text])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(similarity, 3)

if __name__ == "__main__":
    input_text = input("Enter a sentence or paragraph:\n> ")
    predicted = generate_next_sentence(input_text)
    is_grammatical, issues = grammar_check(predicted)
    relevance = relevance_score(input_text, predicted)

    print("\n🔹 Input:", input_text)
    print("🔸 Predicted Next Sentence:", predicted)
    print("✅ Grammar Check:", "Passed" if is_grammatical else "Failed")
    print("📊 Relevance Score:", relevance)
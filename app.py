import torch
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# check if GPU is available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "./trained_model" # path  to the saved model 
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recipe Generator</title>
        <script>
            async function generateRecipe() {
                const ingredients = document.getElementById('ingredients').value;
                const response = await fetch('/generate_recipe', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ ingredients: ingredients })
                });
                const data = await response.json();
                document.getElementById('recipe').innerText = data.recipe;
            }
        </script>
    </head>
    <body>
        <h1>Recipe Generator</h1>
        <textarea id="ingredients" rows="4" cols="50" placeholder="Enter ingredients"></textarea><br>
        <button onclick="generateRecipe()">Generate Recipe</button>
        <h2>Generated Recipe:</h2>
        <p id="recipe"></p>
    </body>
    </html>
    '''


@app.route("/generate_recipe", methods=["POST"])
def generate_recipe():
    data = request.json
    ingredients = data.get("ingredients")
    if not ingredients:
        return jsonify({"error": "No ingredients provided"}), 400

    # generate recipe
    input_text = f"Ingredients: {ingredients}\nRecipe:"
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = inputs.ne(tokenizer.pad_token_id).long()
    
    outputs = model.generate(
        inputs,
        no_repeat_ngram_size = 2,
        attention_mask=attention_mask,
        max_length=250,
        num_return_sequences=1,
        temperature = 0.7, # adjusts randomness
    )

    recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # post process 
    try:
        i = recipe.index('"')
        print('i',i)

        left_text = recipe[:i] 

        return jsonify({"recipe": left_text})
    except ValueError:
        return jsonify({"recipe": recipe})


if __name__ == "__main__":
    app.run(debug=True)

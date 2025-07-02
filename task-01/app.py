from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

model_name = "./my_gpt2_model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route("/", methods=["GET", "POST"])
def index():
    generated_text = ""
    if request.method == "POST":
        prompt = request.form["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.8
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return render_template("index.html", generated_text=generated_text)

if __name__ == "__main__":
    app.run(debug=True)

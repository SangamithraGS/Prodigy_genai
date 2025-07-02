from flask import Flask, render_template, request
import random

app = Flask(__name__)

# Load your corpus into memory once
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

words = text.split()

# Build the Markov chain dictionary
markov_chain = {}

for i in range(len(words)-1):
    word = words[i]
    next_word = words[i+1]

    if word in markov_chain:
        markov_chain[word].append(next_word)
    else:
        markov_chain[word] = [next_word]

@app.route("/", methods=["GET", "POST"])
def index():
    generated_text = ""

    if request.method == "POST":
        start = request.form["start"]
        num_words = 30  # or any length you want

        current_word = start.strip()

        output_words = [current_word]

        for _ in range(num_words):
            if current_word in markov_chain:
                next_word = random.choice(markov_chain[current_word])
                output_words.append(next_word)
                current_word = next_word
            else:
                # Pick a random new word to continue
                current_word = random.choice(list(markov_chain.keys()))
                output_words.append(current_word)

        generated_text = " ".join(output_words)

    return render_template("index.html", generated_text=generated_text)


if __name__ == "__main__":
    app.run(debug=True)

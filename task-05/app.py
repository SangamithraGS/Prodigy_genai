from flask import Flask, request, send_file, render_template
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms

# Import your run_style_transfer() and model setup
# from your own code files here
from nst_module import run_style_transfer

app = Flask(__name__)

# Define transforms
loader = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

@app.route("/", methods=["GET", "POST"])
def home():
    output_image_path = None
    if request.method == "POST":
        # Get uploaded files
        content_file = request.files["content"]
        style_file = request.files["style"]

        # Open as PIL
        content_img = Image.open(content_file.stream).convert("RGB")
        style_img = Image.open(style_file.stream).convert("RGB")

        # Convert to tensors
        content_tensor = loader(content_img).unsqueeze(0)
        style_tensor = loader(style_img).unsqueeze(0)

        # Run your NST optimization
        output_tensor = run_style_transfer(content_tensor, style_tensor)

        # Convert to PIL image
        output_img = transforms.ToPILImage()(output_tensor.squeeze().clamp(0,1))

        # Save result to BytesIO
        img_io = BytesIO()
        output_img.save(img_io, format="PNG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/png")

    return render_template("index.html", output_image=output_image_path)

if __name__ == "__main__":
    app.run(debug=True)

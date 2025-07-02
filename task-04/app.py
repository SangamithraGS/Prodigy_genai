import os
from flask import Flask, request, render_template, send_file
from PIL import Image
import torch
from torchvision import transforms
from io import BytesIO

from models.unet_generator import UNetGenerator

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = UNetGenerator().to(device)
generator.load_state_dict(torch.load("pix2pix_generator.pth", map_location=device))
generator.eval()

# ✅ THIS is the missing part!
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

to_pil = transforms.ToPILImage()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("index.html", error="Please upload a file.")

        file = request.files["file"]
        img = Image.open(file.stream).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = generator(input_tensor)[0].cpu()

        output_image = to_pil(output_tensor.clamp(0, 1))

        # Save input and output images
        input_path = os.path.join("static", "uploads", file.filename)
        output_path = os.path.join("static", "outputs", f"output_{file.filename}")

        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        img.save(input_path)
        output_image.save(output_path)

        return render_template(
            "index.html",
            input_image=input_path,
            output_image=output_path,
            error=None,
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

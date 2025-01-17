from flask import Flask, render_template, request
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt
import io
from PIL import Image

# Initialisation de l'application Flask
app = Flask(__name__)

# Charger le modèle Stable Diffusion
torch.cuda.empty_cache()
model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cpu")

@app.route("/", methods=["GET", "POST"])
def index():
    image_data = None
    if request.method == "POST":
        prompt = request.form["prompt"]
        # Générer l'image avec le prompt de l'utilisateur
        image = pipe(prompt, width=1000, height=1000).images[0]

        # Sauvegarder l'image en mémoire
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        image_data = img_io.read()

    return render_template("index.html", image_data=image_data)

if __name__ == "__main__":
    app.run(debug=True)

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text = clip.tokenize(["The Ego vehicle is nearing an intersection and theres no traffic light",
                    "The Ego vehicle is nearing an intersection and traffic light is green",
                    "The Ego vehicle follows the traffic ahead",
                    "The road is clear ahead",
                    "The Ego vehicle deviates to avoid slow vehicle/obstacles and moves straight",
                    "The vehicle ahead slows down",
                    "Pedestrian or Vehicle cutting in the path ahead",
                    "The Ego vehicle is nearing an intersection and traffic light is red",
                    "The Ego vehicle joins the traffic moving to the right",
                    "A right turn coming ahead",
                    "The traffic moves from the left to right side at the intersection",
                    "The traffic moves from the right to left side at the intersection",
                    "The road is clear ahead on the right lane",
                    "No Speeding vehicle on the right lane is coming from the rear right side",
                    "A left turn coming ahead",
                    "The road is clear ahead on the left lane",
                    "No Speeding vehicle on the left lane is coming from the rear left side"]).to(device)

import pdb;pdb.set_trace()

with torch.no_grad():

    text_features = model.encode_text(text)
    torch.save(text_features,"text_feat.pt")
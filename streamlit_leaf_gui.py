
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


health_classes = ['Disease', 'Dry', 'Healthy']
health_model_path = "leaf_model.pth"


health_model = models.resnet18(pretrained=False)
health_model.fc = nn.Linear(health_model.fc.in_features, len(health_classes))
health_model.load_state_dict(torch.load(health_model_path, map_location=device))
health_model.to(device)
health_model.eval()

health_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


disease_classes = [
    'Apple_Scab_Leaf', 'Apple_leaf', 'Apple_rust_leaf', 'Bell_pepper_leaf',
    'Bell_pepper_leaf_spot', 'Blueberry_leaf', 'Cherry_leaf',
    'Corn_Gray_leaf_spot', 'Corn_leaf_blight', 'Corn_rust_leaf', 'Peach_leaf',
    'Potato_leaf_early_blight', 'Potato_leaf_late_blight', 'Raspberry_leaf',
    'Soyabean_leaf', 'Squash_Powdery_mildew_leaf', 'Strawberry_leaf',
    'Tomato_Early_blight_leaf', 'Tomato_Septoria_leaf_spot', 'Tomato_leaf',
    'Tomato_leaf_bacterial_spot', 'Tomato_leaf_late_blight', 'Tomato_leaf_mosaic_virus',
    'Tomato_leaf_yellow_virus', 'Tomato_mold_leaf', 'Tomato_two_spotted_spider_mites_leaf',
    'grape_leaf', 'grape_leaf_black_rot'
]

disease_keywords = ["blight", "rust", "scab", "spot", "mildew", "virus", "mold", "mites", "black_rot"]

disease_model_path = "D:/iot/best_plantdoc_model.pth"

disease_model = models.efficientnet_b0(weights=None)
disease_model.classifier[1] = torch.nn.Linear(disease_model.classifier[1].in_features, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=device))
disease_model.to(device)
disease_model.eval()

disease_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


st.set_page_config(page_title="🌿 Leaf Health & Disease Detector", layout="centered")
st.title("🌿 Leaf Health & Disease Detector")
st.write("Upload a leaf image and the model will predict its health status.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

   
    input_tensor = health_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = health_model(input_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        health_pred = health_classes[pred.item()]
        health_conf = conf.item()*100

    st.subheader("Leaf Health Prediction:")
    st.write(f"**Status:** {health_pred}")
    st.write(f"**Confidence:** {health_conf:.2f}%")

    if health_pred == "Disease":
        st.warning("The leaf appears unhealthy.")
        if st.button("Check Specific Disease"):
       
            disease_tensor = disease_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                disease_out = disease_model(disease_tensor)
                disease_probs = F.softmax(disease_out, dim=1)
                conf2, pred2 = torch.max(disease_probs, 1)
                disease_pred = disease_classes[pred2.item()]
                disease_conf = conf2.item()*100

            st.subheader("Specific Disease Prediction:")
            st.write(f"**Disease:** {disease_pred}")
            st.write(f"**Confidence:** {disease_conf:.2f}%")

 
            if any(k.lower() in disease_pred.lower() for k in disease_keywords):
                st.info(" This is a disease-affected leaf. Consider treatment options.")
    elif health_pred == "Healthy":
        st.success("The leaf appears healthy.")
    else:
        st.info("The leaf seems dry.")

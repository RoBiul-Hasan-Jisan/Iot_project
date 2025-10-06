import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


leaf_model_path = r"D:\iot\leaf_model.pth"
specific_model_path = r"D:\iot\best_plantdoc_model.pth"

# Load full models
leaf_model = torch.load(leaf_model_path, map_location=device)
leaf_model.eval()

specific_model = torch.load(specific_model_path, map_location=device)
specific_model.eval()


leaf_classes = ["Healthy", "Dry", "Unhealthy"]  # for leaf_model
specific_classes = [
    "Alternaria leaf spot","Brown spot","Gray spot","Pepper_bell_Bacterial_spot",
    "Potato_Early_blight","Potato_Late_blight","Rust","Tomato_Target_Spot",
    "Tomato_Tomato_mosaic_virus","Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_Late_blight",
    "Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot","Tomato_Spider_mites_Two_spotted_spider_mite"
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


def predict_leaf(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = leaf_model(input_tensor)
        _, pred = torch.max(outputs, 1)
        class_name = leaf_classes[pred.item()]
    
    return class_name

def predict_specific(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = specific_model(input_tensor)
        _, pred = torch.max(outputs, 1)
        class_name = specific_classes[pred.item()]
    
    return class_name

def generate_report(image_path):
    status = predict_leaf(image_path)
    report = f"📝 Leaf Health Report\n====================\n📁 File: {image_path}\n\n🌿 Status: {status}\n💡 Recommendation:\n"
    
    if status == "Healthy":
        report += "- Leaf is healthy. Maintain normal care.\n"
    elif status == "Dry":
        report += "- Leaf is dry. Check irrigation and humidity.\n"
    else:
        # Unhealthy: check specific disease
        disease = predict_specific(image_path)
        report += f"- Leaf is affected by {disease}. Take proper treatment.\n"
    
    report += "===================="
    return report


root = tk.Tk()
root.title("Leaf Disease Detection")
root.geometry("750x700")
root.configure(bg="#f5f5f5")

y
img_label = tk.Label(root, bg="#f5f5f5")
img_label.pack(pady=20)


result_text = tk.Text(root, height=12, width=90, font=("Helvetica", 12))
result_text.pack(pady=10)


image_list = []
current_index = 0

def upload_images():
    global image_list, current_index
    files = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not files:
        return
    image_list = list(files)
    current_index = 0
    show_next_image()

def show_next_image():
    global current_index
    if current_index >= len(image_list):
        messagebox.showinfo("Done", "All images have been processed.")
        return
    file_path = image_list[current_index]

    # Display image
    img = Image.open(file_path)
    img = img.resize((300,300))
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

    # Generate and show report
    report = generate_report(file_path)
    result_text.delete('1.0', tk.END)
    result_text.insert(tk.END, report)

def next_image():
    global current_index
    current_index += 1
    if current_index < len(image_list):
        show_next_image()
    else:
        messagebox.showinfo("Done", "No more images.")
        current_index = 0

def cancel_images():
    global image_list, current_index
    image_list = []
    current_index = 0
    result_text.delete('1.0', tk.END)
    img_label.configure(image=None)
    messagebox.showinfo("Cancelled", "Image processing has been cancelled.")


upload_btn = tk.Button(root, text="Upload Leaf Images", command=upload_images,
                       width=30, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
upload_btn.pack(pady=10)

next_btn = tk.Button(root, text="Next Image", command=next_image,
                     width=30, bg="#2196F3", fg="white", font=("Helvetica", 12, "bold"))
next_btn.pack(pady=10)

cancel_btn = tk.Button(root, text="Cancel", command=cancel_images,
                       width=30, bg="#f44336", fg="white", font=("Helvetica", 12, "bold"))
cancel_btn.pack(pady=10)

root.mainloop()

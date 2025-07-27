import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json
import os

# Load pre-trained model
model = tf.keras.applications.MobileNetV2(weights="imagenet")
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

# Load custom labels
custom_labels_file = "custom_labels.json"
if os.path.exists(custom_labels_file):
    with open(custom_labels_file, "r") as f:
        custom_labels = json.load(f)
else:
    custom_labels = {}

def update_custom_labels(original_name, new_name):
    custom_labels[original_name] = new_name
    with open(custom_labels_file, "w") as f:
        json.dump(custom_labels, f, indent=2)

def resolve_label(predicted_name):
    if predicted_name in custom_labels:
        return custom_labels[predicted_name]
    else:
        print(f"AI guessed: {predicted_name}")
        response = input("Is this correct? (y/n): ").strip().lower()
        if response == "y":
            return predicted_name
        else:
            new_name = input("What should it be called?: ").strip()
            link = input(f"Is this the same as '{predicted_name}'? (y/n): ").strip().lower()
            if link == "y":
                update_custom_labels(predicted_name, new_name)
            return new_name

def classify_image(img_array):
    predictions = model.predict(img_array)
    decoded = decode_predictions(predictions, top=3)[0]
    return decoded

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main():
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture image for analysis, or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        cv2.imshow("Object Recognizer - Press SPACE to scan", frame)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            img_array = preprocess_frame(frame)
            predictions = classify_image(img_array)

            for i, (_, name, confidence) in enumerate(predictions):
                print(f"{i+1}. {name} ({confidence*100:.2f}%)")

            chosen = int(input("Choose best match (1-3): ")) - 1
            predicted_name = predictions[chosen][1]
            final_label = resolve_label(predicted_name)

            print(f"✅ Final label: {final_label}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

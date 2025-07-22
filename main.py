import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === 1. Função de carregamento e preprocessamento ===
def load_data(dataset_path, magnification, img_size):
    X, y = [], []
    regex_label = r"[A-Z]+_([A-Z])_[A-Z]+-\d{2}-[A-Z\d]+-(\d+)-\d+\.png"
    num_files = 0

    for root, dirs, files in os.walk(dataset_path):
        if os.path.basename(root) == magnification:
            for file in files:
                if file.endswith('.png'):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, img_size)
                    img = img.astype('float32') / 255.0
                    X.append(img)

                    match = re.search(regex_label, file)
                    if match:
                        label = match.group(1)
                        y.append(1 if label == "M" else 0)
                        num_files += 1

    print(f"Total images processed: {num_files}")
    return np.array(X), np.array(y)

# === 2. Definição da CNN ===
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === Funções adicionais ===
def show_correct_predictions(X_test, y_test, y_pred):
    correct_idx = np.where(y_test == y_pred.reshape(-1))[0]
    benign = [i for i in correct_idx if y_test[i] == 0]
    malign = [i for i in correct_idx if y_test[i] == 1]

    plt.figure(figsize=(10, 4))
    if benign:
        plt.subplot(1, 2, 1)
        plt.imshow(X_test[benign[0]])
        plt.title("Exemplo Acerto: Benigno")
        plt.axis("off")
    if malign:
        plt.subplot(1, 2, 2)
        plt.imshow(X_test[malign[0]])
        plt.title("Exemplo Acerto: Maligno")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def segment_image(img):
    gray = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmented = img.copy()
    cv2.drawContours(segmented, contours, -1, (0, 255, 0), 1)
    return segmented

def save_segmented_image(original_img, label, output_path):
    segmented_img = segment_image(original_img)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(original_img)
    axs[0].set_title("Imagem Original")
    axs[0].axis("off")

    axs[1].imshow(segmented_img)
    axs[1].set_title("Imagem Segmentada")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# === 3. Execução principal ===
if __name__ == "__main__":
    dataset_path = "histology_slides"
    magnification = "400X"
    img_size = (224, 224)

    print("[INFO] Carregando as imagens...")
    X, y = load_data(dataset_path, magnification, img_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Construindo o modelo...")
    model = build_model((img_size[0], img_size[1], 3))

    print("[INFO] Treinando o modelo...")
    datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, zoom_range=0.2)
    datagen.fit(X_train)

    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        epochs=10,
                        validation_data=(X_test, y_test))

    print("[INFO] Avaliando o modelo...")
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))

    labels = ['Benigno', 'Maligno']
    conf = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()

    # Exibir exemplos corretos e segmentados
    print("[INFO] Exibindo exemplos corretamente classificados...")
    show_correct_predictions(X_test, y_test, y_pred)

    print("[INFO] Segmentando imagem de exemplo...")
    output_dir = "segmentados"
    os.makedirs(output_dir, exist_ok=True)

    correct_idx = np.where(y_test == y_pred.reshape(-1))[0]
    benign = [i for i in correct_idx if y_test[i] == 0]
    malign = [i for i in correct_idx if y_test[i] == 1]

    # Salvar imagem de exemplo benigno
    if benign:
        save_segmented_image(X_test[benign[0]], "Benigno", os.path.join(output_dir, "benigno_segmentado.png"))

    # Salvar imagem de exemplo maligno
    if malign:
        save_segmented_image(X_test[malign[0]], "Maligno", os.path.join(output_dir, "maligno_segmentado.png"))

    print(f"[INFO] Imagens salvas em: {output_dir}")

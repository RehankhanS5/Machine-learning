import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
image_size = (64, 64)
train_cat_dir = "C:/Users/hp/Documents/training_set/training_set/cats"
train_dog_dir = "C:/Users/hp/Documents/training_set/training_set/dogs"
test_cat_dir = "C:/Users/hp/Documents/test_set/test_set/cats"
test_dog_dir = "C:/Users/hp/Documents/test_set/test_set/dogs"

def extract_hog(img, image_size=(64, 64)):
    img = cv2.resize(img, image_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(gray,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   visualize=False)
    return features

def load_images_with_hog(folder, label):
    data = []
    labels = []
    original_images = []
    for filename in tqdm(os.listdir(folder), desc=f"Loading {folder}"):
        try:
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is not None:
                img_resized = cv2.resize(img, image_size)
                hog_features = extract_hog(img_resized)
                data.append(hog_features)
                labels.append(label)
                original_images.append(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
    return data, labels, original_images

X_train_cat, y_train_cat, img_train_cat = load_images_with_hog(train_cat_dir, 0)
X_train_dog, y_train_dog, img_train_dog = load_images_with_hog(train_dog_dir, 1)
X_test_cat, y_test_cat, img_test_cat = load_images_with_hog(test_cat_dir, 0)
X_test_dog, y_test_dog, img_test_dog = load_images_with_hog(test_dog_dir, 1)


X_train = np.array(X_train_cat + X_train_dog)
y_train = np.array(y_train_cat + y_train_dog)
X_test = np.array(X_test_cat + X_test_dog)
y_test = np.array(y_test_cat + y_test_dog)
img_test = img_test_cat + img_test_dog  


print("\nTraining SVM...")
svm = LinearSVC(max_iter=10000)
svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))


def plot_predictions(images, X_test, y_test, num_images=8):
    plt.figure(figsize=(15, 6))
    indices = np.random.choice(len(X_test), num_images, replace=False)

    for i, idx in enumerate(indices):
        features = X_test[idx].reshape(1, -1)
        pred = svm.predict(features)[0]
        label = y_test[idx]
        title = f"Pred: {'Dog' if pred == 1 else 'Cat'}\nTrue: {'Dog' if label == 1 else 'Cat'}"

        plt.subplot(2, num_images // 2, i + 1)
        plt.imshow(images[idx])
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
plot_predictions(img_test, X_test, y_test, num_images=6)

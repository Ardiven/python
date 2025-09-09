import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Set random seed untuk reproducibility
np.random.seed(42)


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        """
        Neural Network dengan multiple hidden layers

        Args:
            input_size: Jumlah input features
            hidden_sizes: List berisi ukuran setiap hidden layer
            output_size: Jumlah output neurons (1 untuk binary classification)
            learning_rate: Learning rate untuk training
        """
        self.learning_rate = learning_rate
        self.layers = []

        # Inisialisasi weights dan biases untuk setiap layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.layers.append({'weight': weight, 'bias': bias})

    def sigmoid(self, x):
        """Sigmoid activation function"""
        # Clip x untuk mencegah overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return (x > 0).astype(float)

    def forward_propagation(self, X):
        """Forward propagation through the network"""
        self.activations = [X]
        current_input = X

        # Forward pass through hidden layers (menggunakan ReLU)
        for i in range(len(self.layers) - 1):
            z = np.dot(current_input, self.layers[i]['weight']) + self.layers[i]['bias']
            current_input = self.relu(z)
            self.activations.append(current_input)

        # Output layer (menggunakan sigmoid untuk binary classification)
        z_output = np.dot(current_input, self.layers[-1]['weight']) + self.layers[-1]['bias']
        output = self.sigmoid(z_output)
        self.activations.append(output)

        return output

    def backward_propagation(self, X, y, output):
        """Backward propagation to update weights and biases"""
        m = X.shape[0]  # Number of samples

        # Calculate error for output layer
        error = output - y
        errors = [error]

        # Calculate errors for hidden layers
        for i in range(len(self.layers) - 1, 0, -1):
            if i == len(self.layers) - 1:
                # Output layer error (sudah dihitung di atas)
                continue
            else:
                # Hidden layer error
                error = np.dot(errors[0], self.layers[i]['weight'].T) * self.relu_derivative(self.activations[i])
                errors.insert(0, error)

        # Update weights and biases
        for i in range(len(self.layers)):
            if i == len(self.layers) - 1:
                # Output layer (sigmoid)
                gradient = np.dot(self.activations[i].T, errors[i]) / m
            else:
                # Hidden layers (ReLU)
                gradient = np.dot(self.activations[i].T, errors[i]) / m

            bias_gradient = np.mean(errors[i], axis=0, keepdims=True)

            # Update parameters
            self.layers[i]['weight'] -= self.learning_rate * gradient
            self.layers[i]['bias'] -= self.learning_rate * bias_gradient

    def train(self, X, y, epochs=1000, verbose=True):
        """Train the neural network"""
        self.costs = []

        for epoch in range(epochs):
            # Forward propagation
            output = self.forward_propagation(X)

            # Calculate cost (binary cross-entropy)
            cost = -np.mean(y * np.log(output + 1e-15) + (1 - y) * np.log(1 - output + 1e-15))
            self.costs.append(cost)

            # Backward propagation
            self.backward_propagation(X, y, output)

            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")

    def predict(self, X):
        """Make predictions"""
        output = self.forward_propagation(X)
        return (output > 0.5).astype(int)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.forward_propagation(X)


def generate_synthetic_dataset(n_samples=2000):
    """
    Generate synthetic dataset untuk deteksi manusia vs bukan manusia

    Features yang disimulasikan:
    1. Tinggi (cm)
    2. Berat (kg)
    3. Rasio tinggi/berat
    4. Suhu tubuh (Celsius)
    5. Detak jantung (bpm)
    6. Kecepatan gerakan (km/h)
    7. Ukuran kepala (relatif terhadap tubuh)
    8. Fleksibilitas sendi (0-1)
    """

    # Inisialisasi arrays
    features = []
    labels = []

    # Generate data untuk manusia (label = 1)
    n_humans = n_samples // 2

    for i in range(n_humans):
        # Tinggi manusia (150-200 cm)
        height = np.random.normal(170, 15)
        height = np.clip(height, 150, 200)

        # Berat manusia berdasarkan tinggi dengan variasi
        ideal_weight = (height - 100) * 0.9
        weight = np.random.normal(ideal_weight, 10)
        weight = np.clip(weight, 45, 120)

        # Rasio tinggi/berat
        height_weight_ratio = height / weight

        # Suhu tubuh manusia (36-37.5°C)
        body_temp = np.random.normal(36.8, 0.3)
        body_temp = np.clip(body_temp, 36.0, 37.5)

        # Detak jantung manusia (60-100 bpm saat istirahat)
        heart_rate = np.random.normal(75, 10)
        heart_rate = np.clip(heart_rate, 60, 100)

        # Kecepatan gerakan manusia (0-15 km/h)
        movement_speed = np.random.exponential(3)
        movement_speed = np.clip(movement_speed, 0, 15)

        # Ukuran kepala relatif (manusia ~0.12-0.15 dari tinggi)
        head_size = np.random.normal(0.135, 0.01)
        head_size = np.clip(head_size, 0.12, 0.15)

        # Fleksibilitas sendi manusia (0.6-0.9)
        flexibility = np.random.normal(0.75, 0.08)
        flexibility = np.clip(flexibility, 0.6, 0.9)

        features.append([height, weight, height_weight_ratio, body_temp,
                         heart_rate, movement_speed, head_size, flexibility])
        labels.append(1)

    # Generate data untuk bukan manusia (label = 0)
    n_non_humans = n_samples - n_humans

    for i in range(n_non_humans):
        # Tinggi bervariasi lebih ekstrem
        height = np.random.choice([
            np.random.normal(50, 20),  # Hewan kecil
            np.random.normal(300, 100),  # Hewan besar/objek
            np.random.normal(120, 30)  # Robot/objek sedang
        ])
        height = np.clip(height, 10, 500)

        # Berat tidak mengikuti pola manusia
        weight = np.random.exponential(50)
        weight = np.clip(weight, 1, 1000)

        # Rasio tinggi/berat
        height_weight_ratio = height / weight

        # Suhu tubuh bervariasi ekstrem
        body_temp = np.random.choice([
            np.random.normal(25, 5),  # Objek/robot
            np.random.normal(39, 2),  # Hewan
            np.random.normal(15, 10)  # Objek dingin
        ])
        body_temp = np.clip(body_temp, 0, 50)

        # Detak jantung bervariasi atau nol
        heart_rate = np.random.choice([
            0,  # Objek mati
            np.random.normal(150, 50),  # Hewan dengan detak cepat
            np.random.normal(30, 10)  # Hewan dengan detak lambat
        ])
        heart_rate = np.clip(heart_rate, 0, 300)

        # Kecepatan gerakan bervariasi ekstrem
        movement_speed = np.random.exponential(8)
        movement_speed = np.clip(movement_speed, 0, 100)

        # Ukuran kepala tidak proporsional
        head_size = np.random.uniform(0.05, 0.3)

        # Fleksibilitas berbeda
        flexibility = np.random.uniform(0.1, 1.0)

        features.append([height, weight, height_weight_ratio, body_temp,
                         heart_rate, movement_speed, head_size, flexibility])
        labels.append(0)

    return np.array(features), np.array(labels).reshape(-1, 1)


# Generate dataset
print("Generating synthetic dataset...")
X, y = generate_synthetic_dataset(2000)

# Feature names untuk visualisasi
feature_names = ['Tinggi (cm)', 'Berat (kg)', 'Rasio Tinggi/Berat',
                 'Suhu Tubuh (°C)', 'Detak Jantung (bpm)',
                 'Kecepatan (km/h)', 'Ukuran Kepala', 'Fleksibilitas']

print(f"Dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Manusia: {np.sum(y)} samples, Bukan manusia: {len(y) - np.sum(y)} samples")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

# Inisialisasi dan train neural network
# Arsitektur: 8 input -> 16 hidden -> 8 hidden -> 4 hidden -> 1 output
print("\nInitializing Neural Network...")
print("Architecture: 8 -> 16 -> 8 -> 4 -> 1")

nn = NeuralNetwork(
    input_size=8,
    hidden_sizes=[16, 8, 4],  # 3 hidden layers
    output_size=1,
    learning_rate=0.1
)

print("\nTraining Neural Network...")
nn.train(X_train_scaled, y_train, epochs=1000, verbose=True)

# Make predictions
print("\nMaking predictions...")
train_predictions = nn.predict(X_train_scaled)
test_predictions = nn.predict(X_test_scaled)

train_proba = nn.predict_proba(X_train_scaled)
test_proba = nn.predict_proba(X_test_scaled)

# Evaluate performance
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"\nResults:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification report
print("\nClassification Report (Test Set):")
print(classification_report(y_test, test_predictions,
                            target_names=['Bukan Manusia', 'Manusia']))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Training cost
axes[0, 0].plot(nn.costs)
axes[0, 0].set_title('Training Cost Over Time')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Cost')
axes[0, 0].grid(True)

# 2. Confusion Matrix
cm = confusion_matrix(y_test, test_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bukan Manusia', 'Manusia'],
            yticklabels=['Bukan Manusia', 'Manusia'],
            ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix (Test Set)')
axes[0, 1].set_ylabel('True Label')
axes[0, 1].set_xlabel('Predicted Label')

# 3. Prediction probabilities distribution
axes[1, 0].hist(test_proba[y_test.flatten() == 0], alpha=0.7, label='Bukan Manusia', bins=20)
axes[1, 0].hist(test_proba[y_test.flatten() == 1], alpha=0.7, label='Manusia', bins=20)
axes[1, 0].set_title('Prediction Probability Distribution')
axes[1, 0].set_xlabel('Prediction Probability')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Feature importance (berdasarkan mean absolute weight dari first layer)
first_layer_weights = np.abs(nn.layers[0]['weight']).mean(axis=1)
feature_importance = first_layer_weights / np.sum(first_layer_weights)

bars = axes[1, 1].bar(range(len(feature_names)), feature_importance)
axes[1, 1].set_title('Feature Importance (First Layer Weights)')
axes[1, 1].set_xlabel('Features')
axes[1, 1].set_ylabel('Relative Importance')
axes[1, 1].set_xticks(range(len(feature_names)))
axes[1, 1].set_xticklabels(feature_names, rotation=45, ha='right')

# Add value labels on bars
for bar, importance in zip(bars, feature_importance):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{importance:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Test dengan beberapa contoh
print("\n" + "=" * 50)
print("TESTING DENGAN CONTOH SPESIFIK")
print("=" * 50)

# Contoh manusia
human_example = np.array([[175, 70, 175 / 70, 36.8, 72, 5, 0.135, 0.75]])
human_example_scaled = scaler.transform(human_example)
human_pred = nn.predict_proba(human_example_scaled)

print(f"\nContoh Manusia:")
print(f"Features: {human_example[0]}")
print(f"Prediction probability: {human_pred[0][0]:.4f}")
print(f"Predicted class: {'Manusia' if human_pred[0][0] > 0.5 else 'Bukan Manusia'}")

# Contoh bukan manusia (robot)
robot_example = np.array([[180, 150, 180 / 150, 25, 0, 10, 0.08, 0.3]])
robot_example_scaled = scaler.transform(robot_example)
robot_pred = nn.predict_proba(robot_example_scaled)

print(f"\nContoh Robot:")
print(f"Features: {robot_example[0]}")
print(f"Prediction probability: {robot_pred[0][0]:.4f}")
print(f"Predicted class: {'Manusia' if robot_pred[0][0] > 0.5 else 'Bukan Manusia'}")

# Contoh hewan
animal_example = np.array([[100, 30, 100 / 30, 38.5, 120, 25, 0.2, 0.9]])
animal_example_scaled = scaler.transform(animal_example)
animal_pred = nn.predict_proba(animal_example_scaled)

print(f"\nContoh Hewan:")
print(f"Features: {animal_example[0]}")
print(f"Prediction probability: {animal_pred[0][0]:.4f}")
print(f"Predicted class: {'Manusia' if animal_pred[0][0] > 0.5 else 'Bukan Manusia'}")

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Neural Network Architecture: 8 input -> 16 -> 8 -> 4 -> 1 output")
print(f"Total layers: {len(nn.layers)} (3 hidden + 1 output)")
print(f"Activation functions: ReLU (hidden), Sigmoid (output)")
print(f"Training samples: {len(X_train_scaled)}")
print(f"Test samples: {len(X_test_scaled)}")
print(f"Final test accuracy: {test_accuracy:.4f}")
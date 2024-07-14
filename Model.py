import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf

# Generate synthetic data for IT operations
def generate_it_operations_data():
    num_samples = 1000
    data = {
        'timestamp': pd.date_range(start='1/1/2022', periods=num_samples, freq='h'),
        'cpu_usage': np.random.randint(0, 100, num_samples),
        'memory_usage': np.random.randint(0, 100, num_samples),
        'disk_io': np.random.randint(0, 500, num_samples),
        'network_io': np.random.randint(0, 1000, num_samples),
        'uptime_hours': np.random.randint(0, 10000, num_samples),
        'num_processes': np.random.randint(0, 200, num_samples),
        'issue': np.random.choice([0, 1], num_samples, p=[0.9, 0.1])
    }
    df = pd.DataFrame(data)
    df.to_csv('it_operations_data.csv', index=False)

# Generate synthetic data for network monitoring
def generate_network_data():
    num_samples = 1000
    data = {
        'timestamp': pd.date_range(start='1/1/2022', periods=num_samples, freq='h'),
        'latency': np.random.randint(1, 100, num_samples),
        'packet_loss': np.random.randint(0, 10, num_samples),
        'throughput': np.random.randint(50, 1000, num_samples),
        'jitter': np.random.randint(1, 50, num_samples),
        'issue': np.random.choice([0, 1], num_samples, p=[0.95, 0.05])
    }
    df = pd.DataFrame(data)
    df.to_csv('network_data.csv', index=False)

# Generate the data files
generate_it_operations_data()
generate_network_data()

# Load and preprocess data
def load_and_preprocess_data(file_name):
    data = pd.read_csv(file_name)
    data = data.drop(columns=['timestamp'])
    X = data.drop(columns=['issue']).values
    y = data['issue'].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Build and train a TensorFlow model
def build_and_train_model(X_train, y_train):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    return model, history

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    return accuracy, y_pred

# Plot training history with enhanced visualization
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot training & validation accuracy values
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', marker='o')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    # Plot training & validation loss values
    ax2.plot(history.history['loss'], label='Training Loss', color='blue', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='orange', marker='o')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# IT Operations Model
X_train, X_test, y_train, y_test = load_and_preprocess_data('it_operations_data.csv')
it_operations_model, it_operations_history = build_and_train_model(X_train, y_train)
it_operations_model.save('it_operations_model.keras')
accuracy, y_pred = evaluate_model(it_operations_model, X_test, y_test)
print(f"Test Accuracy: {accuracy}")
print(f'Classification Report:\n{classification_report(y_test, y_pred, zero_division=1)}')
plot_training_history(it_operations_history)

# Network Monitoring Model
X_train, X_test, y_train, y_test = load_and_preprocess_data('network_data.csv')
network_monitoring_model, network_monitoring_history = build_and_train_model(X_train, y_train)
network_monitoring_model.save('network_monitoring_model.keras')
accuracy, y_pred = evaluate_model(network_monitoring_model, X_test, y_test)
print(f"Test Accuracy: {accuracy}")
print(f'Classification Report:\n{classification_report(y_test, y_pred, zero_division=1)}')
plot_training_history(network_monitoring_history)

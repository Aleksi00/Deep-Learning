# Import necessary libraries
import tensorflow.keras.datasets as tfd
import tensorflow as tf
import tensorflow
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load dataset
train_data, test_data = tfd.cifar100.load_data(label_mode="coarse")
(x_train, y_train), (x_test, y_test) = train_data, test_data
x_train, x_test = np.mean(x_train, axis=3), np.mean(x_test, axis=3)

# Load perturbed test set
dict = pickle.load(open("cifar20_perturb_test.pkl", "rb"))
x_perturb, y_perturb = dict["x_perturb"], dict["y_perturb"]
x_perturb = np.mean(x_perturb, axis=3)

# Normalize data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_perturb = x_perturb.astype('float32') / 255.0

print(x_train.shape)
print(x_perturb.shape)
print(y_train.shape)
print(y_perturb.shape)

# a)
num_classes = 20
y_train_encoded = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test_encoded = tensorflow.keras.utils.to_categorical(y_test, num_classes)
y_perturb_encoded = tensorflow.keras.utils.to_categorical(y_perturb, num_classes)  

# Creating a stratified validation set
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
    x_train, y_train_encoded, test_size=0.2, stratify=y_train_encoded, random_state=42)

# Ensure the shapes for confirmation
print("Shapes:")
print("Training Set:", x_train_split.shape, y_train_split.shape)
print("Validation Set:", x_val_split.shape, y_val_split.shape)
print("Test Set:", x_test.shape, y_test_encoded.shape)
print("Perturbed Test Set:", x_perturb.shape, y_perturb_encoded.shape)

# b)
# Define a function to create and train a CNN architecture
def create_and_train_cnn(model, x_train, y_train, x_val, y_val):
    # Compile the model
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Early stopping based on validation loss
    early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    
    # Train the model
    history = model.fit(x_train, y_train, epochs = 3, batch_size=32, 
                        validation_data=(x_val, y_val), callbacks=[early_stopping])
    
    return history
# Define five different CNN architectures
architectures = [
    # Architecture 1
    tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(32, activation='relu'),
        tensorflow.keras.layers.Dense(num_classes, activation='softmax')
    ]),
    
    # Architecture 2
    tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),
        tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(32, activation='relu'),
        tensorflow.keras.layers.Dense(num_classes, activation='softmax')
    ]),
    
    # Architecture 3
    tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),
        tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(128, activation='relu'),
        tensorflow.keras.layers.Dropout(0.3),
        tensorflow.keras.layers.Dense(num_classes, activation='softmax')
    ]),
    
    # Architecture 4
    tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),
        tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(32, activation='relu'),
        tensorflow.keras.layers.Dense(num_classes, activation='softmax')
    ]),
    
    # Architecture 5
    tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),
        tensorflow.keras.layers.Conv2D(32, (5, 5), activation='relu'),
        tensorflow.keras.layers.Conv2D(32, (5, 5), activation='relu'),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(32, activation='relu'),
        tensorflow.keras.layers.Dense(num_classes, activation='softmax')
    ])
]

# Train and evaluate each architecture
results = []
for i, model in enumerate(architectures):
    print(f"Training Architecture {i + 1}")
    history = create_and_train_cnn(model, x_train_split, y_train_split, x_val_split, y_val_split)
    results.append(history.history)

# List of architecture descriptions for table header
architecture_descriptions = [
    "Architecture 1",
    "Architecture 2",
    "Architecture 3",
    "Architecture 4",
    "Architecture 5"
]

# Create a table to report metrics
print("Architecture\t\t\t\t\t\tTrain Acc\tVal Acc\tTrain Loss\tVal Loss")
print("=" * 100)

for i, history in enumerate(results):
    train_acc = history['accuracy'][-1]
    val_acc = history['val_accuracy'][-1]
    train_loss = history['loss'][-1]
    val_loss = history['val_loss'][-1]
    
    print(f"{architecture_descriptions[i]:<75}\t{train_acc:.5f}\t\t{val_acc:.5f}\t{train_loss:.5f}\t\t{val_loss:.5f}")

# c)
# Define different regularization configurations
regularization_configs = [
    {"name": "No Regularization", "dropout_rate": 0.0, "l2_weight": 0.0},
    {"name": "Dropout 0.3", "dropout_rate": 0.3, "l2_weight": 0.0},
    {"name": "Dropout 0.5", "dropout_rate": 0.5, "l2_weight": 0.0},
    {"name": "L2 Regularization 0.001", "dropout_rate": 0.0, "l2_weight": 0.001},
    {"name": "L2 Regularization 0.01", "dropout_rate": 0.0, "l2_weight": 0.01}
]


# Train and evaluate models with different regularization settings
results_regularization = []
for config in regularization_configs:
    print(f"Training Model with {config['name']}")
    model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),
        tensorflow.keras.layers.Conv2D(32, (5, 5), activation='relu'),
        tensorflow.keras.layers.Conv2D(32, (5, 5), activation='relu'),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(32, activation='relu', kernel_regularizer = tensorflow.keras.regularizers.l2(config['l2_weight'])),
        tensorflow.keras.layers.Dropout(config['dropout_rate']),
        tensorflow.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(x_train_split, y_train_split, epochs=3, batch_size=32, 
                        validation_data=(x_val_split, y_val_split),
                        callbacks=[tensorflow.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    
    results_regularization.append((config['name'], history.history))

# d)
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Model creation function for the final model
def create_final_model(num_classes, l2_weight, dropout_rate):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Hyperparameters for the final model
config = {
    'l2_weight': 0.001,  # Placeholder values, adjust these based on your experimentation
    'dropout_rate': 0.3   # Placeholder values, adjust these based on your experimentation
}

# Create and train the final model on the entire training set
final_model = create_final_model(num_classes, config['l2_weight'], config['dropout_rate'])
history_final = final_model.fit(x_train, y_train_encoded, epochs=3, batch_size=32, validation_data=(x_val_split, y_val_split), verbose=1)

# Evaluate the final model on the test set
test_loss, test_accuracy = final_model.evaluate(x_test, y_test_encoded)
print(f"Final Test Accuracy: {test_accuracy}")

# Compute and display the confusion matrix
y_pred = final_model.predict(x_test)
conf_matrix = confusion_matrix(np.argmax(y_test_encoded, axis=1), np.argmax(y_pred, axis=1))
print("Confusion Matrix:")
print(conf_matrix)

# e)
config = {
    'l2_weight': 0.001,  # Placeholder values, adjust these based on your experimentation
    'dropout_rate': 0.3   # Placeholder values, adjust these based on your experimentation
}
def create_final_model(num_classes, l2_weight, dropout_rate):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
final_model = create_final_model(num_classes, config['l2_weight'], config['dropout_rate'])
history_final = final_model.fit(x_train, y_train_encoded, epochs=1, batch_size=32, validation_data=(x_val_split, y_val_split), verbose=1)

def evaluate_on_perturbed_test_set(model, x_test, y_test_encoded, x_perturb, y_perturb_encoded):
    # Evaluate the model on the test set and perturbed test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test_encoded)
    perturb_test_loss, perturb_test_accuracy = model.evaluate(x_perturb, y_perturb_encoded)
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Perturbed Test Accuracy: {perturb_test_accuracy}")

# Approach 1: Regularization and Dropout
def apply_regularization_and_dropout(model, l2_weight=0.001, dropout_rate=0.3):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_regularizer = tf.keras.regularizers.l2(l2_weight)
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.rate = dropout_rate
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Approach 2: Batch Normalization
def apply_batch_normalization(model):
    modified_model = tf.keras.models.Sequential()
    for layer in model.layers:
        modified_model.add(layer)
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            modified_model.add(tf.keras.layers.BatchNormalization())
    modified_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return modified_model

# Approach 3: Data Augmentation
def apply_data_augmentation(x_train, y_train):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        # Add more augmentation layers as needed
    ])
    augmented_x_train = data_augmentation(x_train)
    return augmented_x_train, y_train

# Apply different approaches and evaluate their performance
approaches = {
    "Baseline": final_model,
    "Regularization and Dropout": apply_regularization_and_dropout(final_model, l2_weight=0.001, dropout_rate=0.3),
    "Batch Normalization": apply_batch_normalization(final_model),
}

for approach_name, modified_model in approaches.items():
    print(f"Evaluating Model with {approach_name}")
    modified_model.fit(x_train_split, y_train_split, epochs=1, batch_size=32, validation_data=(x_val_split, y_val_split), verbose=0)
    evaluate_on_perturbed_test_set(modified_model, x_test, y_test_encoded, x_perturb, y_perturb_encoded)
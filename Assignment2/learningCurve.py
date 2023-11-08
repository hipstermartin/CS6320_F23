import matplotlib.pyplot as plt

# Define the training and validation accuracy for each epoch
training_accuracy = [0.530125, 0.58475, 0.618125, 0.642875, 0.66575]
validation_accuracy = [0.02875, 0.03625, 0.1025, 0.06625, 0.04625]

# Define the epochs
epochs = range(1, 6)

# Create the plot
plt.figure(figsize=(10,5))
plt.plot(epochs, training_accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs, validation_accuracy, 'ro-', label='Validation Accuracy')

# Add titles and labels
plt.title('Learning Curve for FFNN Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

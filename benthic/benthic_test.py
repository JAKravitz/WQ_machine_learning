import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from imblearn.over_sampling import SMOTE
import smogn


xdata = pd.read_csv('/Users/jakravit/Desktop/SWIPE_test_copy/emit_rrs.csv',index_col=0)
ydata = pd.read_csv('/Users/jakravit/Desktop/SWIPE_test_copy/BOA_inputs_case2n500.csv',index_col=0)
# targets = ['benthic_coral_gfx', 'benthic_algae_gfx', 'benthic_sediment_gfx', 'benthic_seagrass_gfx', 'benthic_bleachedCoral_gfx']
# data['class'] = np.argmax(data[targets].values, axis=1)


#%%
# get inputs
X  = data.filter(regex='^[0-9]')
# get outputs
y = data[targets]
# Normalize input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#%% class imbalance
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# Calculate class weights
def weighted_mse(y_true, y_pred, class_weights):
    # Compute the squared error for each output
    squared_error = tf.square(y_true - y_pred)
    
    # Weight the squared error by the class weights
    weighted_squared_error = squared_error * class_weights
    
    # Compute the mean of the weighted squared error
    loss = tf.reduce_mean(weighted_squared_error)
    return loss

# calc class weights
# Calculate the sum of each class column
class_sums = y_train.sum(axis=0)
# Calculate the total sum
total_sum = class_sums.sum()
# Calculate the class weights as the inverse of the class proportions
class_weights = total_sum / class_sums

# wrapper function
def get_weighted_mse(class_weights):
    def wrapped_weighted_mse(y_true, y_pred):
        return weighted_mse(y_true, y_pred, class_weights)
    return wrapped_weighted_mse

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(5, activation='softmax')  # 5 output classes
])

# Compile the model
model.compile(optimizer='adam', loss=get_weighted_mse(class_weights), metrics=['accuracy'])

# Display the model architecture
model.summary()

# Step 5: Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, )

# Step 6: Evaluate and predict using the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Make predictions
predictions = model.predict(X_test)

# Obtain the predicted classes
predicted_classes = np.argmax(predictions, axis=1)

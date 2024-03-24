import numpy as np
from unet import unet_model as unet1
from unet_model import unet_model as unet2

# Load the models
model1 = unet1()
model2 = unet2()

# Print the model summaries
# print("Model 1 Summary:")
# model1.summary()
print("\nModel 2 Summary:")
model2.summary()

# # Compare the output for the same input
# input_data = np.random.random((1, 128, 128, 3))  # replace with your input shape
# output1 = model1.predict(input_data)
# output2 = model2.predict(input_data)

# # Check if the outputs are close enough
# if np.allclose(output1, output2):
#     print("The models produce the same output for the given input.")
# else:
#     print("The models produce different outputs for the given input.")

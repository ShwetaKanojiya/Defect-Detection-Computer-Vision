import numpy as np
import tensorflow as tf
import cv2

def get_img_array(img_path, size):
    """
    Loads an image from a path, resizes it, and pre-processes 
    it exactly as the MobileNetV2 feature extractor expects.
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0) # Add batch dimension
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap showing the pixels in the input image 
    that had the most impact on the final classification decision.
    """
    # Create a model that outputs the target layer and the final prediction
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            # For our binary classification with sigmoid, preds shape is (1, 1).
            # The output probability targets class index 1.
            # We compute gradients for this node output.
            class_channel = preds[:, 0]
        else:
            class_channel = preds[:, pred_index]

    # Gradient of the top predicted class w.r.t. the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Pool the gradients over all axes leaving out the channel dimension
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map by its importance (pooled gradient)
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_gradcam_overlay(img_path, model, size=(224, 224), last_conv_layer_name="out_relu", alpha=0.4):
    """
    Utility end-to-end wrapper for Streamlit application to visualize 
    Grad-CAM directly on the original input image.
    """
    try:
        img_array = get_img_array(img_path, size=size)
        
        # Determine the last convolutional layer dynamically if the default name isn't found
        layer_names = [layer.name for layer in model.layers]
        if last_conv_layer_name not in layer_names:
            for layer in reversed(model.layers):
                # Ensure we only pick 2D Convolutional layers
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        
        # Load the original image with OpenCV
        original_img = cv2.imread(img_path)
        # Convert BGR to RGB (OpenCV default is BGR, Streamlit needs RGB)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Resize heatmap and apply coloring
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Superimpose the heatmap with alpha blending
        superimposed_img = heatmap * alpha + original_img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
        
        return superimposed_img
    except Exception as e:
        print(f"[Warn] Grad-CAM error (this can happen if model structure changes): {e}")
        return None

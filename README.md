
# Deepfake Detector [CYBERSCAN

Hey there! Welcome to my Deepfake Detector  for Deep Learning Project. 
I’ve built a Convolutional Neural Network (CNN) to figure out if an image is real or a sneaky deepfake. This wasn’t a straight shot—I went through a few ups and downs to get here—but I ended up with a model that’s accurate, tough against new images, and even has a cool web app to play with. Let me walk you through what I did, how it works, and how you can try it out yourself!

## What’s This All About?
The goal was to  training a deep learning model to spot deepfakes. I started with a massive pile of images, tweaked my approach a few times, and landed on a dataset of 85,714 images—half real, half fake. After splitting them up, training a CNN, and adding a Streamlit web interface, I’ve got something that not only works well but also robust.

## 88.6% test accuracy,
Robustness across diverse fake sources (Stable Diffusion,Groke, other AI-generated faces, etc.),
Streamlit web app for interactive testing

 Here’s the story of how I got there.

## The Dataset: Building the Foundation

### Where It Came From
I wanted a dataset that was big, balanced, and diverse—something my model could really learn from.
Here’s what I gathered: The total dataset size : 85,714 images of fakes and reals.

42,857Real Images: Pulled from all over—think stock photos, public datasets, and my own images. anything with variety in faces, lighting, and angles.

42,857 Fake Images: 
15,000 from www.Kaggle.com .
10,000 I cooked up myself using Stable Diffusion .
15,000 from https://this-person-does-not-exist.com/en#google_vignette —those AI-generated faces are wild!
2,857 from own generating through multiple LLms

## Splitting the data || 
I shuffled everything and split it into:
Train: 60,000 images (30,000 real, 30,000 fake) – 70%
Test: 12,857 images (6,429 real, 6,429 fake) – 15%
Validation: 12,857 images (6,429 real, 6,429 fake) – 15%

## Making It Ready
### Preprocessing:
Resizing: All images scaled to 256×256 pixels.
Augmentation: Rotation (±20°), shifts (20%), zoom, shear, horizontal flip.
Normalization: Pixel values rescaled to [0, 1].

## Why This Way?
A 50-50 real/fake split avoids bias, and the variety (Kaggle, Stable Diffusion, LLms and Websites etc.) means the model sees all kinds of fakes—not just one type. Plus, ~86k  images is a very good data: big enough to train well, not so big it crashes my laptop!

### The Model: My CNN Adventure
The Journey
* Round 1: I started with a huge dataset and a basic CNN. Accuracy was decent, but it flopped on new images—classic overfitting.

* Round 2: Scaled down to 20,000 images (10,000 real, 10,000 fake), tweaked some settings, and added dropout. It generalized better, but still tripped up on some fakes.

* Round 3 (The Winner): Went big again with approx 86K images, mixed from different sources, and built a smarter CNN. This one nailed it—accurate and solid on unseen data.
  
## The Model: Architecture & Training
### Final CNN Architecture

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)),
    layers.BatchNormalization(),  # Stabilizes training
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),  # Deeper features
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),  # Expanded capacity
    layers.Dropout(0.5),  # Regularization
    layers.Dense(1, activation='sigmoid')  # Binary output
])

## Training Process

1.Optimizer: Adam (lr=0.001) with ReduceLROnPlateau (reduces LR by 50% if validation loss stalls).
2.Early Stopping: Halts training if no improvement for 5 epochs (prevents overfitting).
3.Epochs: 15 (best validation accuracy: 89.3%).

4.Layers: Three sets of convolution (32, 64, 128 filters) and pooling to spot patterns, a dense layer (128) to think hard, and dropout (0.5) . 5.Sigmoid at the end for a 0-1 real/fake call.
6.Training: Used Adam optimizer, binary_crossentropy loss (perfect for yes/no questions), and watched accuracy. Ran for up to 25 epochs but stopped early if validation loss stalled for 3 rounds (early stopping).

How It Learned
Batch Size: 32 images at a time.
Augmentation: Only on training data to mimic real-world chaos.
Result: Saved as my_model.h5 

## Performance
### Test Accuracy: 
Achieved 88.6% accuracy on our test set of 12,857 unseen images (balanced real/fake distribution)

### Robustness Assessment:
1.Cross-Dataset Validation
Tested on 50 completely new images (25 real, 25 fake) from external sources
Maintained high accuracy of 94%, demonstrating strong generalization
2.Real-World Simulation
Created custom test cases by face-swapping personal photos of myself and friends
Successfully identified manipulated content with consistent accuracy


### Try It Yourself: The Web App
I wrapped this all up in a Streamlit app—it’s like a little playground for the model!
What It Does: Upload an image, and it tells you “Real” or “Fake” with a confidence score.


What I Learned & What’s Next
This was a ride! Early models were too picky or too blend to new fakes. Mixing datasets, adding dropout, and tweaking the CNN fixed that. If I had more time, I’d:
aim to develop an advanced detection system capable of analyzing not just images, but also audio and video content. Such a tool would serve as a crucial resource for both the public and legal systems, helping distinguish authentic media from AI-generated manipulations. In an era where synthetic content is becoming increasingly sophisticated, this technology would support truth verification and contribute to upholding justice in digital spaces.

##  How to Run

Clone the repo 

1.Install dependencies:

```bash
  pip install tensorflow streamlit pillow
```
2.Run the app:
```bash
  streamlit run streamlit.py
```


import matplotlib.pyplot as plt
import numpy as np
from utils.Caption import get_caption
from utils.Emotions import get_emotions


def get_label(image):
    caption = get_caption(image)

    pred = get_emotions(image, caption)

    emotions = ['Excitement', 'Sadness', 'Amusement', 'Disgust', 'Awe', 'Contentment', 'Fear', 'Anger']
    probabilities = pred
    print(pred)
    max_idx = np.argmax(probabilities)

    # Create color list where all bars are one color, and the max bar is another color
    bar_colors = ['skyblue' if i != max_idx else 'orange' for i in range(len(emotions))]

    # Create bar chart
    fig, ax = plt.subplots()
    ax.bar(emotions, probabilities, color=bar_colors, width=0.5)
    ax.set_ylabel('Probability')
    ax.set_title('Emotion Prediction Probabilities')

    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()

    return caption, fig

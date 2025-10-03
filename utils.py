# utils.py

##% Import Libraries
import os
import matplotlib.pyplot as plt
import pandas as pd

##% Save Plot Function
def save_plot(filename):
    """Save a matplotlib plot to the outputs folder."""
    os.makedirs("outputs", exist_ok=True)
    clean_filename = filename.replace("outputs/", "")
    save_path = os.path.join("outputs", clean_filename)
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")
    plt.close()  # Close the figure to avoid overlap

##% Save Results Function
def save_results(df, filename):
    """Save a DataFrame to a CSV file in the outputs folder."""
    os.makedirs("outputs", exist_ok=True)
    clean_filename = filename.replace("outputs/", "")
    save_path = os.path.join("outputs", clean_filename)
    df.to_csv(save_path, index=False)
    print(f"Results saved as {save_path}")
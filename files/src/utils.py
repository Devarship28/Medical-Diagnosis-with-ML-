import matplotlib.pyplot as plt
import seaborn as sns

def plot_corr(df):
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()

def plot_histograms(df):
    df.hist(bins=20, figsize=(15,10))
    plt.tight_layout()
    plt.show()
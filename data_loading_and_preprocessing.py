import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(path='creditcard.csv', visualize=True):
    df = pd.read_csv(path)

    if visualize:
        plt.figure(figsize=(6,4))
        class_counts = df['Class'].value_counts()
        sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
        plt.title("Class Distribution")
        plt.xlabel("Class (0: Normal, 1: Fraud)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        important_features = ['V1', 'V2', 'V3', 'V4', 'Amount']
        df[important_features + ['Class']].hist(bins=30, figsize=(12,10), layout=(3, 2), color='#4682B4')
        plt.suptitle("Distribution of Selected Features")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8,6))
        corr = df[important_features + ['Class']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6,6))
        sns.scatterplot(x='V1', y='V2', hue='Class', data=df.sample(1000), palette='Set1', alpha=0.6)
        plt.title("V1 vs V2 Scatter Plot (Sample)")
        plt.tight_layout()
        plt.show()

        for feature in ['V4', 'Amount']:
            plt.figure(figsize=(6,4))
            sns.boxplot(x='Class', y=feature, data=df, palette="Set2")
            plt.title(f"{feature} vs Class")
            plt.tight_layout()
            plt.show()

    features = df.drop(columns=['Class'])
    labels = df['Class']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, labels.values

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle


# Function to train and evaluate the knn classifier
def train_and_evaluate(input_file:str , output_folder):
    # Load the data 
    data = np.load(input_file ,  allow_pickle=True).item()
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    # Initialize and train
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    with open(output_folder, 'wb') as file:
        pickle.dump(knn, file)

    # Predict and evaluate
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

if __name__ == '__main__':
    accuracy_scores = []
    # Loop over each fold and perform training and evaluation
    for fold in range(1, 6):
        accuracy = train_and_evaluate(f'././data/train_test_fold_{fold}.npy', f'././results/split_{fold}/knn_model.pkl')
        accuracy_scores.append(accuracy)

    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)

    print(f"Mean Accuracy: {mean_accuracy}")
    print(f"Standard Deviation: {std_accuracy}")
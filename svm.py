from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

dataset = np.load('time_series.npy', allow_pickle=True).item()
labels = dataset["labels"]
time_series = dataset["timeSeries"]

X_flattened = time_series.reshape(time_series.shape[0], -1)

#StratifiedKFold with 5 splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# List to store accuracy scores
accuracy_scores = []


for train_index, test_index in skf.split(X_flattened, labels):
    X_train, X_test = X_flattened[train_index], X_flattened[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# la moyenne et l'ecart-type
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)


print(f"Mean Accuracy: {mean_accuracy}")
print(f"l'ecart-type: {std_accuracy}")
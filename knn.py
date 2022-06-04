from sklearn.neighbors import KNeighborsClassifier

# Model Building
def build_knn(data = None, target = None):
    if data is not None and target is not None:
        knn = KNeighborsClassifier()
        knn.fit(data, target)
        return knn
    else:
        print('Data and Target is Missing')
        return None

def knn_score(model = None, data = None, target = None):
    print('Calculating Training Accuracy: ')
    train_acc = model.score(data, target)
    print('Training Accuracy: ', train_acc)
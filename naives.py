from sklearn.naive_bayes import GaussianNB

# Model Building
def build_nb(data = None, target = None):
    if data is not None and target is not None:
        nb = GaussianNB()
        nb.fit(data, target)
        return nb
    else:
        print('Data and Target is Missing')
        return None
def nb_score(model = None, data = None, target = None):
    print('Calculating Training Accuracy')
    train_acc = model.score(data ,target)
    print('Training Accuracy: ',train_acc)

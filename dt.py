from sklearn.tree import DecisionTreeClassifier

# Model Building
def build_dt(data = None, target = None):
    if data is not None and target is not None:
        dt = DecisionTreeClassifier()
        dt.fit(data, target)
        return dt
    else:
        print('Data and Target is Missing')
        return None
def dt_score(model = None, data = None, target = None):
    print('Calculating Training Accuracy: ')
    train_acc = model.score(data, target)
    print('Training Accuracy: ', train_acc)

from sklearn.ensemble import RandomForestClassifier

# Model Building
def build_rf(data = None, target = None):
    if data is not None and target is not None:
        rf = RandomForestClassifier()
        rf.fit(data, target)
        return rf
    else:
        print('Data and Target is Missing')
        return None
        
def rf_score(model = None, data = None, target = None):
    print('Calculating Training Accuracy: ')
    train_acc = model.score(data, target)
    print('Training Accuracy: ', train_acc)
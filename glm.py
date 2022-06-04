from sklearn.linear_model import LogisticRegression

# Model Building
def build_glm(data = None, target = None):
    if data is not None and target is not None:
        glm = LogisticRegression()
        glm.fit(data, target)
        return glm
    else:
        print('Data and Target is Missing')
        return None

# Metrics
def glm_score(model = None, data = None, target = None):
    print("Calculating Training Accuracy: ")
    train_acc = model.score(data, target)
    print('Training Accuracy: ', train_acc)
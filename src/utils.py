import pickle

def save_model(model, model_name):
    with open(f'models/best_model_{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

from train import train_model
from utils import save_prediction_sample

if __name__ == "__main__":

    #train the model
    model, train_data, val_data = train_model()

    #generate and save some prediction samples
    print("Saving prediction samples on validation set...")
    save_prediction_sample(model, val_data, "outputs/val")
    print("Saving prediction samples on training set...")
    save_prediction_sample(model, train_data, "outputs/train")

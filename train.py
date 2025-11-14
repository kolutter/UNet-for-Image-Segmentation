import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from dataloader import GirafeDataset
from model import build_unet
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping

# Calculate Dice score
def dice_score(y_true, y_pred, smooth=1e-6):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)  
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Dice loss = 1 - dice_score, so you can minimize it
def dice_loss(y_true, y_pred):
    return 1.0 - dice_score(y_true, y_pred)

# Calculate precision metric 
def precision_score(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = tf.round(K.cast(y_pred, 'float32'))
    true_positives = K.sum(y_true * y_pred)
    predicted_positives = K.sum(y_pred)
    return true_positives / (predicted_positives + K.epsilon())


def train_model():
    # Define paths 
    dataset_path = "./GIRAFE_dataset/Training"
    image_dir = os.path.join(dataset_path, "imagesTr")
    mask_dir = os.path.join(dataset_path, "labelsTr")

    # List all images and masks
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_files]

    # Split into train/val sets, 80% train, 20% val
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    # Wrap data in custom dataset class
    train_data = GirafeDataset(train_imgs, train_masks)
    val_data = GirafeDataset(val_imgs, val_masks)

    # Compile with Adam, dice loss, and metrics
    model = build_unet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=dice_loss,
        metrics=[dice_score, precision_score]
    )

    # save bist model in checkpoints folder
    checkpoint_cb = ModelCheckpoint(
        "checkpoints/unet_model.keras",
        save_best_only=True,
        monitor="val_dice_score",
        mode="max"
    )

    # Early stopping for the case of worsening validation dice score
    early_stop = EarlyStopping(
        monitor="val_dice_score",
        patience=3,
        restore_best_weights=True,
        mode="max"
    )

    # Train the model
    model.fit(
        train_data,
        epochs=40,
        validation_data=val_data,
        callbacks=[checkpoint_cb, early_stop]
    )

    return model, train_data, val_data


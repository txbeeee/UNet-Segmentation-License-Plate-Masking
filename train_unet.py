from pathlib import Path
import tensorflow as tf

IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4

ROOT = Path("/workspace/licenseplate")
TRAIN_JSON = ROOT / "train" / "train.json"
TRAIN_IMAGES = ROOT / "train"
VAL_JSON = ROOT / "valid" / "_annotations.coco.json"
VAL_IMAGES = ROOT / "valid"

def main():
    train_records = load_coco_records(TRAIN_JSON, TRAIN_IMAGES)
    val_records = load_coco_records(VAL_JSON, VAL_IMAGES)
    train_ds = make_segmentation_dataset(train_records, batch_size=BATCH_SIZE, shuffle=True, training=True)
    val_ds = make_segmentation_dataset(val_records, batch_size=BATCH_SIZE, shuffle=False, training=False)
    model = build_unet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    def dice_coef(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    def dice_loss(y_true, y_pred):
        return 1.0 - dice_coef(y_true, y_pred)
    bce = tf.keras.losses.BinaryCrossentropy()
    def bce_dice_loss(y_true, y_pred):
        return bce(y_true, y_pred) + dice_loss(y_true, y_pred)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=bce_dice_loss,
        metrics=[dice_coef]
    )
    model.summary()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    model.save("unet_segmentation.h5")

if __name__ == "__main__":
    main()

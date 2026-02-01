import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(x, filters, name_prefix):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name_prefix}_conv1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name_prefix}_conv2")(x)
    return x

def encoder_block(x, filters, name_prefix):
    c = conv_block(x, filters, name_prefix=name_prefix)
    p = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool")(c)
    return c, p

def decoder_block(x, skip, filters, name_prefix):
    x = layers.UpSampling2D((2, 2), name=f"{name_prefix}_up")(x)
    x = layers.Concatenate(name=f"{name_prefix}_concat")([x, skip])
    x = conv_block(x, filters, name_prefix=name_prefix)
    return x

def build_unet(input_shape=(512, 512, 3)):
    inputs = layers.Input(shape=input_shape, name="input_image")
    c1, p1 = encoder_block(inputs, 64,  "enc1")
    c2, p2 = encoder_block(p1,     128, "enc2")
    c3, p3 = encoder_block(p2,     256, "enc3")
    c4, p4 = encoder_block(p3,     512, "enc4")
    b = conv_block(p4, 1024, name_prefix="bottleneck")
    d4 = decoder_block(b,  c4, 512, name_prefix="dec4")
    d3 = decoder_block(d4, c3, 256, name_prefix="dec3")
    d2 = decoder_block(d3, c2, 128, name_prefix="dec2")
    d1 = decoder_block(d2, c1, 64,  name_prefix="dec1")
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="mask")(d1)
    model = models.Model(inputs=inputs, outputs=outputs, name="UNet")
    return model

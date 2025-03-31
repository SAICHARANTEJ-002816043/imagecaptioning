import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Input, Add

def build_model(vocab_size, max_len):
    # Image feature extractor
    base_model = InceptionV3(weights='imagenet', include_top=False)
    base_model.trainable = False
    img_input = Input(shape=(299, 299, 3), name='image_input')
    img_features = base_model(img_input)
    img_features = tf.keras.layers.GlobalAveragePooling2D()(img_features)
    img_features = Dense(256, activation='relu')(img_features)
    img_features = Dropout(0.5)(img_features)

    # Caption sequence model
    cap_input = Input(shape=(max_len,), name='caption_input')
    cap_embed = Embedding(vocab_size, 256, mask_zero=True)(cap_input)
    cap_lstm = LSTM(256)(cap_embed)

    # Combine and predict
    combined = Add()([img_features, cap_lstm])
    output = Dense(vocab_size, activation='softmax')(combined)

    model = tf.keras.Model(inputs=[img_input, cap_input], outputs=output)
    return model

if __name__ == "__main__":
    model = build_model(vocab_size=5000, max_len=40)
    model.summary()
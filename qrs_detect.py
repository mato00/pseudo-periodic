import tensorflow as tf
tf.keras.backend.set_floatx('float64')

from qrs_net import QRSNet


def qrs_detect(data, model, threshold):
    preds = model.predict(data)
    preds = np.squeeze(preds)
    preds[preds >= threshold] = 1
    preds[preds < threshold] = 0

    return preds

if __name__ == '__main__':
    MODEL_PATH = './model/'
    THRESHOLD = 0.5

    model = QRSNet()
    ckpt_dir = os.path.dirname(MODEL_PATH)
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt_path:
        model.load_weights(ckpt_path)

    preds = qrs_detect(ecg_data, model, THRESHOLD)

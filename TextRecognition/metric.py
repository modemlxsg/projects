import tensorflow as tf
import utils
import numpy as np


class OCR_Accuracy(tf.keras.metrics.Metric):

    def __init__(self, name='ocr_acc'):
        super(OCR_Accuracy, self).__init__(name=name)
        self.total = self.add_weight(
            name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.positive = self.add_weight(
            name='positive', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.decoder = utils.Decoder()
        self.config = utils.load_config()
        self.blank = self.config['lexicon']['blank']

    def update_state(self, y_true, y_pred):
        batch_size = y_pred.shape[0]
        self.total.assign_add(batch_size)

        y_true = tf.sparse.to_dense(y_true, default_value=self.blank)
        y_true = self.decoder.convert2str(y_true)
        y_pred = self.decoder.decode(
            y_pred, 'greedy', default_value=self.blank)
        y_pred = self.decoder.convert2str(y_pred)

        count = np.count_nonzero(np.array(y_true) == np.array(y_pred))
        self.positive.assign_add(count)

    def reset_states(self):
        self.total.assign(0)
        self.positive.assign(0)

    def result(self):
        return self.positive / self.total


if __name__ == "__main__":
    OCR_Accuracy()

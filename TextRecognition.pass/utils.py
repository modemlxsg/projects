import tensorflow as tf
import yaml


def load_config(config_path='config.yaml'):
    config_file = open(config_path, 'r', encoding='utf-8')
    data = config_file.read()
    config_file.close()
    config = yaml.full_load(data)
    return config


def load_lexicon():
    lexicon = []
    with open('./lexicon.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            lexicon.append(line)

    return lexicon


class Decoder:

    def __init__(self):
        self.config = load_config()
        self.lexicon = load_lexicon()
        self.blank = self.config['lexicon']['blank']

    def convert2str(self, decoded):
        batch_size = decoded.shape[0]
        strs = []
        for i in range(batch_size):
            txt = ""
            for code in decoded[i]:
                if code == self.blank:
                    break
                txt += self.lexicon[code]
            strs.append(txt)

        return strs

    def decode(self, sequence, mode, default_value):
        inputs = tf.constant(sequence)
        inputs = tf.transpose(inputs, perm=(1, 0, 2))
        sequence_length = tf.constant([inputs.shape[0]] * inputs.shape[1])

        if mode == 'greedy':
            decoded, _ = tf.nn.ctc_greedy_decoder(
                inputs, sequence_length, merge_repeated=True)
        elif mode == 'beam_search':
            decoded, _ = tf.nn.ctc_greedy_decoder(inputs, sequence_length)
        else:
            raise Exception('Mode Error')

        decoded = tf.sparse.to_dense(
            decoded[0], default_value=default_value).numpy()
        return decoded

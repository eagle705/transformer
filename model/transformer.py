from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
from pprint import pprint
from model.ops import MultiHeadAttention



class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, config):
        super(EncoderLayer, self).__init__(name='EncoderLayer')

        self.config = config
        self.vocab_size = config['vocab_size']
        self.embed_dim = self.config['embed_dim'] # d_model
        self.head_num = self.config['head_num'] # h # split_embed_dim * head_num == embed_dim
        self.split_embed_dim = self.config['split_embed_dim'] # dim_k, dim_v # self-attention에는 context vector에 쓰였던 context vector를 위한 attention dim 개념이 없음, 자기 차원끼리 attention을 구하니까 attention을 위한 벡터가 따로 필요없거든
        self.layer_num = config['layer_num'] # N
        self.feed_forward_dim = config['feed_forward_dim'] # dim_ffc

        # Define your layers here.
        self.embed = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim, input_shape=(None,))

        # Multi Head Attention
        self.mha = MultiHeadAttention(self.config)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.position_wise_fc_1 = tf.keras.layers.Dense(units=self.feed_forward_dim, activation='relu')
        self.position_wise_fc_2 = tf.keras.layers.Dense(units=self.embed_dim)

    def add_positional_encoding(self, embed):
        # ref: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/text/transformer.ipynb#scrollTo=1Rz82wEs5biZ
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates

        def positional_encoding(position, d_model):
            angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                    np.arange(d_model)[np.newaxis, :],
                                    d_model)

            # apply sin to even indices in the array; 2i
            sines = np.sin(angle_rads[:, 0::2])
            # apply cos to odd indices in the array; 2i+1
            cosines = np.cos(angle_rads[:, 1::2])
            pos_encoding = np.concatenate([sines, cosines], axis=-1)
            pos_encoding = pos_encoding[np.newaxis, ...]

            return tf.cast(pos_encoding, dtype=tf.float32)

        pos_encoding = positional_encoding(self.vocab_size, self.embed_dim)
        seq_len = tf.shape(embed)[1]
        return embed + pos_encoding[:, :seq_len, :]


    def position_wise_fc(self, vector):
        out = self.position_wise_fc_1(vector) # (batch, seq, dim_ffc)
        out = self.position_wise_fc_2(out) # (batch, seq, model_dim)
        return out


    def sub_layer(self, x, training=False, padding_mask=None):
        out_1, attention_weight = self.mha(x, K = x, V = x, mask=padding_mask, flag="encoder_mask") # 첫번째 인자 Q를 input으로 인식함
        out_1 = self.dropout1(out_1, training=training)
        out_2 = self.layer_norm_1(out_1 + x)
        out_3 = self.position_wise_fc(out_2)
        out_3 = self.dropout2(out_3, training=training)
        out_4 = self.layer_norm_2(out_2 + out_3)

        return out_4, attention_weight


    def call(self, inputs, training=False, mask=None):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        # print("inputs: ", inputs)
        self.maxlen = tf.shape(inputs)[1]

        x = self.embed(inputs)  # (batch, seq, word_embedding_dim)
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        x = self.add_positional_encoding(x)
        attention_weights = {}
        for i in range(self.layer_num):
            x, attention_weight = self.sub_layer(x, training, mask)
            attention_weights['encoder_layer{}_block1'.format(i + 1)] = attention_weight


        return x, attention_weights # encoder_output, attention weights

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, config):
        super(DecoderLayer, self).__init__(name='DecoderLayer')

        self.config = config
        self.vocab_size = config['vocab_size']
        self.embed_dim = self.config['embed_dim'] # d_model
        self.head_num = self.config['head_num'] # h
        self.split_embed_dim = self.config['split_embed_dim'] # d_k, d_v
        self.layer_num = config['layer_num']  # N
        self.feed_forward_dim = config['feed_forward_dim']  # dim_ffc

        # Define your layers here.
        self.embed = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim, input_shape=(None,))

        # Multi Head Attention
        self.look_ahead_mha = MultiHeadAttention(self.config)
        self.mha = MultiHeadAttention(self.config)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.position_wise_fc_1 = tf.keras.layers.Dense(units=self.feed_forward_dim, activation='relu')
        self.position_wise_fc_2 = tf.keras.layers.Dense(units=self.embed_dim)

    def add_positional_encoding(self, embed):
        # ref: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/text/transformer.ipynb#scrollTo=1Rz82wEs5biZ
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates

        def positional_encoding(position, d_model):
            angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                    np.arange(d_model)[np.newaxis, :],
                                    d_model)

            # apply sin to even indices in the array; 2i
            sines = np.sin(angle_rads[:, 0::2])
            # apply cos to odd indices in the array; 2i+1
            cosines = np.cos(angle_rads[:, 1::2])
            pos_encoding = np.concatenate([sines, cosines], axis=-1)
            pos_encoding = pos_encoding[np.newaxis, ...]

            return tf.cast(pos_encoding, dtype=tf.float32)

        pos_encoding = positional_encoding(self.vocab_size, self.embed_dim)
        seq_len = tf.shape(embed)[1]
        return embed + pos_encoding[:, :seq_len, :]


    def position_wise_fc(self, vector):
        out = self.position_wise_fc_1(vector) # (batch, seq, dim_ffc)
        out = self.position_wise_fc_2(out) # (batch, seq, model_dim)
        return out


    def sub_layer(self, x, encoder_ouput, training=False, look_ahead_mask=None, padding_mask=None):
        out_1, attention_weight_lah_mha_in_decoder = self.look_ahead_mha(x, K = x, V = x, mask = look_ahead_mask, flag="look_ahead_mask")
        out_1 = self.dropout1(out_1, training=training)
        out_2 = self.layer_norm_1(out_1 + x)
        out_3, attention_weight_pad_mha_in_decoder = self.mha(out_2, K = encoder_ouput, V = encoder_ouput, mask = padding_mask, flag="padding_mask")
        out_3 = self.dropout2(out_3, training=training)
        out_4 = self.layer_norm_2(out_3 + out_2)
        out_5 = self.position_wise_fc(out_4)
        out_6 = self.layer_norm_3(out_4 + out_5)

        return out_6, attention_weight_lah_mha_in_decoder, attention_weight_pad_mha_in_decoder


    def call(self, inputs, encoder_ouput, training, look_ahead_mask, padding_mask):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        # print("decoder inputs: ", inputs)
        self.maxlen = tf.shape(inputs)[1]

        x = self.embed(inputs)  # (batch, seq, word_embedding_dim)
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))

        x = self.add_positional_encoding(x)
        attention_weights = {}
        for i in range(self.layer_num):
            x, attention_block1, attention_block2 = self.sub_layer(x, encoder_ouput, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = attention_block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = attention_block2

        return x, attention_weights

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


class Transformer(tf.keras.Model):
    def __init__(self, config):
        super(Transformer, self).__init__(name='Transformer')
        self.vocab_size = config['vocab_size']
        self.encoder = EncoderLayer(config)
        self.decoder = DecoderLayer(config)
        self.linear = tf.keras.layers.Dense(self.vocab_size)#, activation='relu')
        self.softmax = tf.keras.layers.Softmax()

    def call(self, encoder_input, decoder_input, training, enc_padding_mask, combined_mask, dec_padding_mask):
        encoder_output, attention_weight_in_encoder = self.encoder(encoder_input, training, enc_padding_mask)
        decoder_output, attention_weight_in_decoder = self.decoder(decoder_input, encoder_output, training, combined_mask, dec_padding_mask) # inputs, encoder_ouput, training, look_ahead_mask, padding_mask
        softmax_prob = self.softmax(self.linear(decoder_output))

        return softmax_prob, attention_weight_in_decoder

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


def main():
    print("Model")


if __name__ == '__main__':
    main()
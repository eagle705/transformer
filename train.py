from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from data_loader import build_vocab, word_to_pad_word_ids, decode_word_ids
from model.transformer import Transformer

print(tf.__version__)
tf.executing_eagerly()
np.set_printoptions(suppress=True)

def main():
    training_dataset = [["안녕하세요, 제 이름은 윤주성입니다", "hello, my name is joosung yoon"],
                        ["저는 텐서플로우를 좋아합니다", "i like tensorflow"]]
    X_y_split = list(zip(*training_dataset))

    X_train_str = list(X_y_split[0]) # ['안녕하세요, 제 이름은 윤주성입니다', '저는 텐서플로우를 좋아합니다']
    y_train_str = list(X_y_split[1]) # ['Hello, my name is joosung Yoon', 'I like TensorFlow']
    print(X_train_str)
    print(y_train_str)

    corpus = []
    corpus.extend(X_train_str)
    corpus.extend(y_train_str)  # ['안녕하세요, 제 이름은 윤주성입니다', '저는 텐서플로우를 좋아합니다', 'Hello, my name is joosung Yoon', 'I like TensorFlow']

    vocab = build_vocab(corpus)
    print(vocab.idx2word)

    max_sequence_len = 13

    X_train, _, _ = word_to_pad_word_ids(text_batch=X_train_str, vocab=vocab, maxlen=max_sequence_len, add_start_end_token=True)
    _, tar_inp, tar_real = word_to_pad_word_ids(text_batch=y_train_str, vocab=vocab, maxlen=max_sequence_len, add_start_end_token=True) # add +1 maxlen for start, end token

    print(X_train) # [[ 5  6  7  8  9 10 11 12 13 14  0  0  0  0  0], [15 16 17 18 19  0  0  0  0  0  0  0  0  0  0]]
    print(tar_inp) # [[20  8 21 22 23 24 25  0  0  0  0  0  0  0  0], [26 27 28  0  0  0  0  0  0  0  0  0  0  0  0]]
    print(tar_real)

    print(decode_word_ids(X_train, vocab))
    # [['안녕/NNG', '하/XSV', '세요/EP+EF', ',/SC', '제/MM', '이름/NNG', '은/JX', '윤주/NNG', '성/XSN', '입니다/VCP+EC', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'],
    # ['저/NP', '는/JX', '텐서플로우/NNP', '를/JKO', '좋아합니다/VV+EC', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']]

    config = {}
    config['vocab_size'] = len(vocab.idx2word)
    config['maxlen'] = max_sequence_len
    config['embed_dim'] = 100
    config['head_num'] = 5
    config['split_embed_dim'] = 20
    config['layer_num'] = 2
    config['feed_forward_dim'] = 100

    # define model
    model = Transformer(config = config)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')  # input label == index of class
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0)) # padding 아닌건 1
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask # 패딩이 아닌 1인 값은 살리고, 패딩인 값인 0인 값은 없앰

        return tf.reduce_mean(loss_)

    def create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions so that we can add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(step_size):
        """
        - decoder에서 각 상태에 대한 self-attention이 inference step에 맞게 future token을 보지 못하게 해야됨
        - 각 step이 소유하고 있는 attention은 step개수 만큼임
        - future token보지 못하게 하려면 각 step에서 future step에 대해서 마스킹 해야함
        - 1 step에서는 나머지 n-1개 masking, 2번째 스텝에서는 앞에 두개 빼고 나머지 n-2개 마스킹
        - 이렇게 하면 역삼각형 모양의 마스킹 매트릭스가 나옴
        - step * step 을 대각선으로 나눈 모양임

        example)
        x = tf.random.uniform((1, 3))
        temp = create_look_ahead_mask(x.shape[1])
        temp:
        <tf.Tensor: id=311521, shape=(3, 3), dtype=float32, numpy=
        array([[ 0.,  1.,  1.],
               [ 0.,  0.,  1.],
               [ 0.,  0.,  0.]], dtype=float32)>

        Special usecase:
         tf.matrix_band_part(input, 0, -1) ==> Upper triangular part.
         tf.matrix_band_part(input, -1, 0) ==> Lower triangular part.
         tf.matrix_band_part(input, 0, 0) ==> Diagonal.
        :param step_size:
        :return:

        """
        mask = 1 - tf.linalg.band_part(tf.ones((step_size, step_size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def create_masks(inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    # 세션 대신 tf.function() decorator로 파이썬 함수를 감싸면, 이 함수를 하나의 그래프로 실행하기 위해 JIT 컴파일함
    # tf.function()을 쓰면 eager mode -> graph mode 되는 것임
    # @tf.function
    def train_step(enc_input, tar_inp, tar_real):
        # tar_inp = label[:, :-1] # remove </s>
        # tar_real = label[:, 1:] # remove <s>

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(enc_input, tar_inp)


        with tf.GradientTape() as tape:
            predictions, attention_weights = model(enc_input, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = loss_function(tar_real, predictions) # masking losses for padding


            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32).numpy()
            print("X_train: ", decode_word_ids(enc_input.numpy(), vocab))
            print("tar_inp: ", decode_word_ids(tar_inp.numpy(), vocab))
            print("tar_real: ", decode_word_ids(tar_real.numpy(), vocab))
            print("result: ", decode_word_ids(predicted_id, vocab))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(tar_real, predictions)

    # @tf.function
    # def test_step(Y_test, label):
    #     predictions = model(Y_test)
    #     t_loss = loss_object(label, predictions)
    #
    #     test_loss(t_loss)
    #     test_accuracy(label, predictions)

    def plot_attention_weights(attention, sentence, result, layer):
        import matplotlib.pyplot as plt
        from matplotlib import font_manager, rc
        # print("font_list: ", font_manager.get_fontconfig_fonts())
        font_name = font_manager.FontProperties(fname='/Library/Fonts/NanumSquareBold.ttf').get_name()
        rc('font', family=font_name)

        fig = plt.figure(figsize=(16, 8))

        sentence, _, _ = word_to_pad_word_ids(text_batch=[sentence], vocab=vocab, maxlen=max_sequence_len, add_start_end_token=True) #tokenizer_pt.encode(sentence)
        attention = tf.squeeze(attention[layer], axis=0)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head + 1)

            # plot the attention weights
            im = ax.matshow(attention[head][:, :], cmap='viridis') # viridis  #plt.cm.Reds # plt.cm.Blues

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(decode_word_ids(sentence, vocab)[0])))
            ax.set_yticks(range(len(decode_word_ids(result, vocab)[0])))

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.set_xticklabels(decode_word_ids(sentence, vocab)[0], fontdict=fontdict, rotation=90)

            ax.set_yticklabels(decode_word_ids(result, vocab)[0], fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head + 1))

        plt.tight_layout()
        plt.show()

    def evaluate(inp_sentence, vocab, max_sequence_len):

        # inference 일때는 굳이 length를 +1 하지 않아도됨
        encoder_input, _, _ = word_to_pad_word_ids(text_batch=[inp_sentence], vocab=vocab, maxlen=max_sequence_len, add_start_end_token=True)
        print("encoder_input: ", encoder_input)

        decoder_input = ['<s>']
        decoder_input = [vocab.word2idx[_] for _ in decoder_input]
        output = tf.expand_dims(decoder_input, 0)
        print("output: ", decode_word_ids(output.numpy(), vocab))

        for i in range(max_sequence_len):

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = model(encoder_input,
                               output,
                               False,
                               enc_padding_mask,
                               combined_mask,
                               dec_padding_mask)

            # select the last word from the seq_len dimension
            print("predicted_id: ", tf.cast(tf.argmax(predictions, axis=-1), tf.int32))
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, vocab.word2idx['</s>']):
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)
            print("output: ", decode_word_ids(output.numpy(), vocab))

        return tf.squeeze(output, axis=0), attention_weights

    def translate(sentence, vocab, max_sequence_len, plot=''):
        result, attention_weights = evaluate(sentence, vocab, max_sequence_len)
        result = [result.numpy()]

        predicted_sentence = decode_word_ids(result, vocab)

        print('Input: {}'.format(sentence))
        print('Predicted translation: {}'.format(predicted_sentence))

        if plot:
            plot_attention_weights(attention_weights, sentence, result, plot)



    ### Training

    EPOCHS = 4000
    BATCH_SIZE = 45

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, tar_inp, tar_real))
    train_ds = train_ds.repeat(EPOCHS).shuffle(1024).batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)


    for step, (X_train_batch, tar_inp, tar_real) in enumerate(train_ds):
        train_step(X_train_batch, tar_inp, tar_real)

        template = 'Step {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(step + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))



    translate("안녕하세요, 제 이름은 윤주성입니다", vocab, max_sequence_len, plot='decoder_layer2_block2')


    model.summary()

if __name__ == '__main__':
    main()
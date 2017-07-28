import helper
import tensorflow as tf
import numpy as np
PAD = 0
EOS = 1
# UNK = 2
# GO  = 3

vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

#Embeddings
embeddings = tf.Variable(tf.truncated_normal([vocab_size, input_embedding_size], mean=0.0, stddev=0.1), dtype=tf.float32)
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)
print(encoder_inputs_embedded)
#encoder
encoder_cell = tf.contrib.rnn.BasicLSTMCell(encoder_hidden_units)
lstm_layers = 4
cell = tf.contrib.rnn.MultiRNNCell([encoder_cell] * lstm_layers)
# If `time_major == True`, this must be a `Tensor` of shape:
#       `[max_time, batch_size, ...]`, or a nested tuple of such
#       elements.
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell,encoder_inputs_embedded,dtype=tf.float32,time_major=True)
del encoder_outputs
print(encoder_final_state)
#decoder
decoder_cell = tf.contrib.rnn.BasicLSTMCell(decoder_hidden_units)
decoder = tf.contrib.rnn.MultiRNNCell([decoder_cell] * lstm_layers)
decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder, decoder_inputs_embedded,
    initial_state=encoder_final_state,
    dtype=tf.float32, time_major=True, scope="plain_decoder",
)
decoder_logits = tf.contrib.layers.fully_connected(decoder_outputs,vocab_size,activation_fn=None,
                                              weights_initializer = tf.truncated_normal_initializer(stddev=0.1),
                                              biases_initializer=tf.zeros_initializer())
# decoder_prediction = tf.argmax(decoder_logits,)
print(decoder_logits)
decoder_prediction = tf.argmax(decoder_logits,2) # 在这一步我突然意识到了axis的含义。。。表明的竟然是在哪个维度上求 argmax。
print(decoder_prediction)
# learn_rate = tf.placeholder(tf.float32)
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)
loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     batch_ = [[6], [3, 4], [9, 8, 7]]
#
#     batch_, batch_length_ = helper.batch(batch_)
#     print('batch_encoded:\n' + str(batch_))
#
#     din_, dlen_ = helper.batch(np.ones(shape=(3, 1), dtype=np.int32),
#                                max_sequence_length=4)
#     print('decoder inputs:\n' + str(din_))
#
#     pred_ = sess.run(decoder_prediction,
#                      feed_dict={
#                          encoder_inputs: batch_,
#                          decoder_inputs: din_,
#                          #             learn_rate:0.1,
#                      })
#     print('decoder predictions:\n' + str(pred_))
#
# print("build graph ok!")

batch_size = 100
batches = helper.random_sequences(length_from=3, length_to=8,
                                   vocab_lower=2, vocab_upper=10,
                                   batch_size=batch_size)
print('head of the batch:')
for seq in next(batches)[:10]:
    print(seq)
def next_feed():
    batch = next(batches)
    encoder_inputs_, _ = helper.batch(batch)
    decoder_targets_, _ = helper.batch(
        [(sequence) + [EOS] for sequence in batch]
    )
    decoder_inputs_, _ = helper.batch(
        [[EOS] + (sequence) for sequence in batch]
    )
    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }
loss_track = []

max_batches = 3001
batches_in_epoch = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
        for batch in range(max_batches):
            fd = next_feed()
            _, l = sess.run([train_op, loss], fd)
            loss_track.append(l)

            if batch == 0 or batch % batches_in_epoch == 0:
                print('batch {}'.format(batch))
                print('  minibatch loss: {}'.format(sess.run(loss, fd)))
                predict_ = sess.run(decoder_prediction, fd)
                for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                    print('  sample {}:'.format(i + 1))
                    print('    input     > {}'.format(inp))
                    print('    predicted > {}'.format(pred))
                    if i >= 2:
                        break
                print()
    except KeyboardInterrupt:
        print('training interrupted')

#matplotlib inline
import matplotlib.pyplot as plt
plt.plot(loss_track)
plt.show()
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))
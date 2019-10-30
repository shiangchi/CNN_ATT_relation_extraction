import numpy as np
np.random.seed(5000)  # for reproducibility

import _pickle as pkl
import gzip
import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.models import Sequential,Model, load_model
from keras_self_attention import SeqSelfAttention
from keras.layers import Dense, Dropout, Activation, Flatten, Add, Merge, Reshape, RepeatVector, Multiply, Subtract, Average
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Bidirectional, LSTM
from keras.utils import np_utils
from gensim.models import word2vec, KeyedVectors
from keras.layers.merge import concatenate
from keras.layers import Embedding, Input

batch_size = 256
nb_filter = 50
filter_length = 5
hidden_dims = 50
nb_epoch = 15
position_dims = 30

print("Load dataset")

file = '../pkl_CDR/CDR_only_single_v2.pkl.gz' # remember to change the output file name 
f = gzip.open(file, 'rb')

yTrain, sentenceTrain, chemTrain , disTrain, positionTrain1, positionTrain2, POSMatrix_train, train_s1_sdp, sim_train_s1, sim_train_s11, sim_train_s12 = pkl.load(f,encoding='bytes')
yTest, sentenceTest, chemTest , disTest, positionTest1, positionTest2, POSMatrix_test, test_s1_sdp, sim_test_s1, sim_test_s11, sim_test_s12 = pkl.load(f,encoding='bytes')

f.close()

max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

n_out = max(yTrain)+1
test_y_cat = np_utils.to_categorical(yTest, n_out)
train_y_cat = np_utils.to_categorical(yTrain, n_out)

f = gzip.open('../pkl_CDR/CDR_single_embeddings.pkl.gz', 'rb')
embeddings = pkl.load(f,encoding='bytes')
f.close()

f_sent = gzip.open('../pkl_CDR/CDR_single_sentence_embedding.pkl.gz', 'rb')
sent_embeddings = pkl.load(f_sent,encoding='bytes')
f.close()

sent_embeddings = sent_embeddings.reshape((-1,700))
print("Sentence Embeddings :",  sent_embeddings.shape)
print("Embeddings: ",embeddings.shape)

# input layer is used to recieve a length of chemTrain.shape[1] number serise
# enbedding layer : input_dim = vocab size , output_dim = embedding dim, input_length = max sentence length. model.output_shape == (None, input_length, embedding dim)
chem_name = Input(shape=(chemTrain.shape[1],), dtype='int32') 
chem_emb = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], input_length=chemTrain.shape[1], trainable=True)(chem_name)

disease_name = Input(shape=(disTrain.shape[1],), dtype='int32')
dis_emb = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], input_length=disTrain.shape[1], trainable=True)(disease_name)

distance1 = Input(shape=(positionTrain1.shape[1],), dtype='int32')
distanceModel1 = Embedding(max_position, position_dims, input_length=positionTrain1.shape[1])(distance1)

distance2 = Input(shape=(positionTrain2.shape[1],), dtype='int32')
distanceModel2 = Embedding(max_position, position_dims, input_length=positionTrain2.shape[1])(distance2)

word = Input(shape=(sentenceTrain.shape[1],), dtype='int32')
word_emb = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], input_length=sentenceTrain.shape[1], trainable=False)(word)

pos = Input(shape=(POSMatrix_train.shape[1],), dtype='int32')
pos_emb = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], input_length=POSMatrix_train.shape[1])(pos)

s1 = Input(shape=(sim_train_s1.shape[1],), dtype='int32')
s1_emb = Embedding(input_dim=sent_embeddings.shape[0], output_dim=sent_embeddings.shape[1], weights=[sent_embeddings], trainable=False)(s1)

s11 = Input(shape=(sim_train_s11.shape[1],), dtype='int32')
s11_emb = Embedding(input_dim=sent_embeddings.shape[0], output_dim=sent_embeddings.shape[1], weights=[sent_embeddings], trainable=False)(s11)

s12 = Input(shape=(sim_train_s12.shape[1],), dtype='int32')
s12_emb = Embedding(input_dim=sent_embeddings.shape[0], output_dim=sent_embeddings.shape[1], weights=[sent_embeddings], trainable=False)(s12)

s1_sdp = Input(shape=(train_s1_sdp.shape[1],), dtype='int32')
s1_sdp_emb = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], trainable=False)(s1_sdp)


bilstm_s1_sdp = Bidirectional(LSTM(5))(s1_sdp_emb) 
bilstm_s1_sdp = RepeatVector(sentenceTest.shape[1])(bilstm_s1_sdp)
word_emb = concatenate([word_emb, bilstm_s1_sdp])

merged = concatenate([word_emb, chem_emb, dis_emb, distanceModel1, distanceModel2, pos_emb])

cnn_5 = Conv1D(
            filters=nb_filter,
            kernel_size=filter_length,
            padding='same',
            activation='relu',
            strides=1)(merged)

cnn_5 = GlobalMaxPooling1D()(cnn_5)
cnn_5 = Dropout(0.4)(cnn_5)

extra_sent_s1 = concatenate([s1_emb, s11_emb, s12_emb],axis=1)
att_sent_s1 = SeqSelfAttention(attention_activation='hard_sigmoid',attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,name='attention_all')(extra_sent_s1)
att_dense = Dense(units=200, activation='relu')(att_sent_s1)
att_dense = Flatten()(att_dense)
cnn = concatenate([cnn_5, att_dense])
cnn = Dense(100, activation='tanh')(cnn)

x = Dense(n_out, activation='softmax',
                kernel_regularizer=regularizers.l2(0.01))(cnn)

#compile

final_modle = Model(inputs=[word, chem_name, disease_name, distance1, distance2, \
    pos, s1_sdp, s1, s11, s12], outputs=[x])

# model.summary()
final_modle.summary()
final_modle.compile(loss='binary_crossentropy',optimizer='Adam', metrics=['accuracy'])

print("Start training")



max_acc = 0
max_acc_ep = 0
# for epoch in xrange(nb_epoch):
for epoch in range(nb_epoch):
    loss = 0 
    accuracy = 0
   
    train_x = [sentenceTrain, chemTrain, disTrain, positionTrain1, positionTrain2, POSMatrix_train, \
        train_s1_sdp, sim_train_s1, sim_train_s11, sim_train_s12] 
    test_x = [sentenceTest, chemTest, disTest, positionTest1, positionTest2, POSMatrix_test, \
        test_s1_sdp, sim_test_s1, sim_test_s11, sim_test_s12]


    final_modle.fit(train_x, train_y_cat, class_weight={0 : 1., 1 : 1.25}, batch_size=batch_size, \
    verbose=True, epochs=10, shuffle=True, validation_data=(test_x, test_y_cat)) # 0:UN 1:CID
    
    # save model 
    final_modle.save('../model/cnn_embedding.h5')

    # evaluate the model
    loss, accuracy = final_modle.evaluate(train_x, train_y_cat, verbose=2)
    print('Train score: %f' % loss)
    print('Train Accuracy: %f' % (accuracy*100))
    pred_test = final_modle.predict(test_x, verbose=0)
    y_classes = pred_test.argmax(axis=-1)


    dctLabels = np.sum(y_classes)
    totalDCTLabels = np.sum(yTest)
    print("")
    print("predictLabels",dctLabels)
    print("totalLabels",totalDCTLabels)
    
    print("sum", np.sum(y_classes == yTest))

    acc =  np.sum(y_classes == yTest) / float(len(yTest))

    res = (epoch,acc)
    if res[1] > max_acc :
        # print(res[1], max_acc)
        max_acc = res[1]
        max_acc_ep = res[0]
        cross = "cnn_across"
        with open("../pred_result"+'/'+cross+'_only_across','w') as fout:    
            for i in range(0,len(y_classes)):
                fout.write(str(y_classes[i]) +'\n')
            fout.close()

        with open("../pred_result"+'/'+cross+'_prob_only_across','w') as fout:    
            for i in range(0,len(pred_test)):
                fout.write(str(pred_test[i]) +'\n')
            fout.close()
            
    # print(epoch)
    print("Test Accuracy: %.4f (max_acc: %.4f, max_acc_ep : %d)" % (acc, max_acc, max_acc_ep))


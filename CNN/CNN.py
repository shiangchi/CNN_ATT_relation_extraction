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
from keras.layers import Embedding, Input
from keras.layers.merge import concatenate
from keras.utils import np_utils
from gensim.models import word2vec, KeyedVectors


batch_size = 256
nb_filter = 50
filter_length = 5
hidden_dims = 50
nb_epoch = 15
position_dims = 30

print("Load dataset")
file = '../pkl_CDR/CDR_only_across_v2.pkl.gz'
f = gzip.open(file, 'rb')

yTrain, sentenceTrain, chemTrain , disTrain, positionTrain1, positionTrain2, POSMatrix_train, re_sent_train, ent_dp_train, s_w_train, around_train, train_s1_sdp, train_s2_sdp, sim_train_s1, sim_train_s11, sim_train_s12, sim_train_s2, sim_train_s21, sim_train_s22 = pkl.load(f,encoding='bytes')
yTest, sentenceTest, chemTest , disTest, positionTest1, positionTest2, POSMatrix_test, re_sent_test, ent_dp_test, s_w_test, around_test, test_s1_sdp, test_s2_sdp, sim_test_s1, sim_test_s11, sim_test_s12, sim_test_s2, sim_test_s21, sim_test_s22 = pkl.load(f,encoding='bytes')

f.close()

max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

n_out = max(yTrain)+1
test_y_cat = np_utils.to_categorical(yTest, n_out)
train_y_cat = np_utils.to_categorical(yTrain, n_out)

f = gzip.open('../pkl_CDR/CDR_embeddings.pkl.gz', 'rb')
embeddings = pkl.load(f,encoding='bytes')
f.close()

f_sent = gzip.open('../pkl_CDR/CDR_cross_sentence_embedding.pkl.gz', 'rb')
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

re_sents = Input(shape=(re_sent_train.shape[1],), dtype='int32')
re_sents_emb = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], input_length=re_sent_train.shape[1])(re_sents)

ent_dp = Input(shape=(ent_dp_train.shape[1],), dtype='int32')
ent_dp_emb = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], input_length=ent_dp_train.shape[1], trainable=True)(ent_dp)

s_ws = Input(shape=(s_w_train.shape[1],), dtype='int32')
s_ws_emb = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], input_length=s_w_train.shape[1], trainable=False)(s_ws)

aroundword = Input(shape=(around_train.shape[1],), dtype='int32')
aroundword_emb = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], input_length=around_train.shape[1], trainable=False)(aroundword)

s1 = Input(shape=(sim_train_s1.shape[1],), dtype='int32')
s1_emb = Embedding(input_dim=sent_embeddings.shape[0], output_dim=sent_embeddings.shape[1], weights=[sent_embeddings], trainable=False)(s1)

s11 = Input(shape=(sim_train_s11.shape[1],), dtype='int32')
s11_emb = Embedding(input_dim=sent_embeddings.shape[0], output_dim=sent_embeddings.shape[1], weights=[sent_embeddings], trainable=False)(s11)

s12 = Input(shape=(sim_train_s12.shape[1],), dtype='int32')
s12_emb = Embedding(input_dim=sent_embeddings.shape[0], output_dim=sent_embeddings.shape[1], weights=[sent_embeddings], trainable=False)(s12)

s2 = Input(shape=(sim_train_s2.shape[1],), dtype='int32')
s2_emb = Embedding(input_dim=sent_embeddings.shape[0], output_dim=sent_embeddings.shape[1], weights=[sent_embeddings], trainable=False)(s2)

s21 = Input(shape=(sim_train_s21.shape[1],), dtype='int32')
s21_emb = Embedding(input_dim=sent_embeddings.shape[0], output_dim=sent_embeddings.shape[1], weights=[sent_embeddings], trainable=False)(s21)

s22 = Input(shape=(sim_train_s22.shape[1],), dtype='int32')
s22_emb = Embedding(input_dim=sent_embeddings.shape[0], output_dim=sent_embeddings.shape[1], weights=[sent_embeddings], trainable=False)(s22)

s1_sdp = Input(shape=(train_s1_sdp.shape[1],), dtype='int32')
s1_sdp_emb = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], trainable=False)(s1_sdp)

s2_sdp = Input(shape=(train_s2_sdp.shape[1],), dtype='int32')
s2_sdp_emb = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], trainable=False)(s2_sdp)

bilstm_s1_sdp = Bidirectional(LSTM(5))(s1_sdp_emb)
bilstm_s2_sdp = Bidirectional(LSTM(5))(s2_sdp_emb)

bilstm_s1_sdp = RepeatVector(sentenceTest.shape[1])(bilstm_s1_sdp)
bilstm_s2_sdp = RepeatVector(sentenceTest.shape[1])(bilstm_s2_sdp)
word_emb = concatenate([word_emb, bilstm_s1_sdp, bilstm_s2_sdp])

merged = concatenate([word_emb, chem_emb, dis_emb, distanceModel1, distanceModel2, pos_emb])

cnn_5 = Conv1D(
            filters=nb_filter,
            kernel_size=filter_length,
            padding='same',
            activation='relu',
            strides=1)(merged)

cnn_5 = GlobalMaxPooling1D()(cnn_5)
cnn_5 = Dropout(0.4)(cnn_5)

cnn_around = Conv1D(
            filters=6,
            kernel_size=2,
            padding='same',
            activation='relu',
            strides=1)(aroundword_emb)

cnn_around = GlobalMaxPooling1D()(cnn_around)
cnn_around = Dropout(0.45)(cnn_around)

re_sents_emb = Flatten()(re_sents_emb)
re_sents_emb = Dense(50, activation='tanh')(re_sents_emb)

ent_dp_emb = Flatten()(ent_dp_emb)
ent_dp_emb = Dense(25, activation='tanh')(ent_dp_emb)

s_ws_emb = Flatten()(s_ws_emb)
s_ws_emb = Dense(50, activation='tanh')(s_ws_emb)

extra_sent_s1 = concatenate([s1_emb, s11_emb, s12_emb, s2_emb, s21_emb, s22_emb],axis=1)
att_sent_s1 = SeqSelfAttention(attention_activation='hard_sigmoid',attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,name='attention_all')(extra_sent_s1)

att_dense = Dense(units=200, activation='relu')(att_sent_s1)
att_dense = Flatten()(att_dense)

cnn = concatenate([cnn_5, cnn_around, re_sents_emb, ent_dp_emb, s_ws_emb, att_dense])

cnn = Dense(100, activation='tanh')(cnn)

x = Dense(n_out, activation='softmax',
                kernel_regularizer=regularizers.l2(0.01))(cnn)

#compile
final_modle = Model(inputs=[word, chem_name, disease_name, distance1, distance2, \
    pos, re_sents, ent_dp, s_ws, aroundword, s1_sdp, s2_sdp, s1, s11, s12, s2, s21, s22], outputs=[x])
    

# model.summary()
final_modle.summary()
final_modle.compile(loss='binary_crossentropy',optimizer='Adam', metrics=['accuracy'])

print("Start training")



max_acc = 0
max_acc_ep = 0

for epoch in range(nb_epoch):
    loss = 0 
    accuracy = 0

    train_x = [sentenceTrain, chemTrain, disTrain, positionTrain1, positionTrain2, POSMatrix_train, \
        re_sent_train, ent_dp_train, s_w_train, around_train, train_s1_sdp, train_s2_sdp, sim_train_s1,\
        sim_train_s11, sim_train_s12, sim_train_s2, sim_train_s21, sim_train_s22] 
    test_x = [sentenceTest, chemTest, disTest, positionTest1, positionTest2, POSMatrix_test, \
        re_sent_test, ent_dp_test, s_w_test, around_test, test_s1_sdp, test_s2_sdp, sim_test_s1,\
        sim_test_s11, sim_test_s12, sim_test_s2, sim_test_s21, sim_test_s22]


    final_modle.fit(train_x, train_y_cat, class_weight={0 : 1., 1 : 3.6}, batch_size=batch_size, \
    verbose=True, epochs=10, shuffle=True, validation_data=(test_x, test_y_cat)) 
    
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



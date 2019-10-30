import re
import numpy as np
import cPickle as pkl
from nltk import FreqDist
import gzip
import codecs
import sys
import time
from comb_sent_emb import comb_emb
from gensim.models import word2vec, KeyedVectors
sys.path.insert(0,'../corpus')

from parser import *
nlp = stanford_nlp() 

outputFilePath = '../pkl_CDR/CDR_only_across_v2.pkl.gz' 
embeddingsPklPath = '../pkl_CDR/CDR_embeddings.pkl.gz'

train_emb = '../corpus/sent_pkl/sentence_embedding_train_v7.pkl'
test_emb = '../corpus/sent_pkl/sentence_embedding_test_v7.pkl'

comb_emb(train_emb, test_emb, 'CDR_cross_sentence_embedding')


embeddingsPath = "../glove.6B.300d.txt"
sentence_embPath = "../corpus/sent_pkl/CDR_cross_sentence_embedding.pkl"
sentence_embPklPath = "../pkl_CDR/CDR_cross_sentence_embedding.pkl.gz"

folder = '../corpus/mult_feat/'
files = [folder+'CDR_TrainingSet.PubTator_new_4_v7_del_same_7', folder+'CDR_TestSet.PubTator_new_4_v7_del_same_7']

labelsMapping = {'UN':0, 
                 'CID':1}

words = {}
sents = {}
maxSentenceLen = [0,0]

labelsDistribution = FreqDist()

distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
minDistance = -60
maxDistance = 60

for dis in xrange(minDistance,maxDistance+1):
    distanceMapping[dis] = len(distanceMapping)

def createMatrices(file, Bioword2Idx, sentenceIndex, maxSentenceLen=200):
    """Creates matrices for the events and sentence for the given file"""
    labels = []
    positionMatrix1 = []
    positionMatrix2 = []
    tokenMatrix = []
    chemMatrix = []
    disMatrix = []
    POSMatrix = []
    retokenMatrix = []
    s_wMatrix = []
    ent_dpM = []
    around_wordM = []
    s1_sdpM = []
    s2_sdpM = []
    Similar_sent1M = []
    Similar_sent11M = []
    Similar_sent12M = []
    Similar_sent2M = []
    Similar_sent21M = []
    Similar_sent22M = []

    with codecs.open(file, encoding="utf-8") as f1:
        inst_lines = f1.readlines()
        for i in range(len(inst_lines)):            
            parts = inst_lines[i].strip('\n').split("\t")
            label = parts[0].strip()
            chem = parts[2].strip()
            dis = parts[4].strip()
            chem_pos1 = parts[6].strip()
            dis_pos2 = parts[7].strip()
            sentence = parts[9].strip()
            POS_tag = parts[10].strip()
            rewrite_sent = parts[11]
            coref_y_n = parts[12]
            ent_dp = parts[13]
            s_w = parts[14] # s_w = similar_word
            c_around = parts[15]
            d_around = parts[16]
            s1_sdp = parts[17]
            s2_sdp = parts[18]
            sim_sent_1 = parts[19]
            sim_sent_11 = parts[20] 
            sim_sent_12 = parts[21]
            sim_sent_2 = parts[22]
            sim_sent_21 = parts[23]
            sim_sent_22 = parts[24]

            chems = chem.split(' ')
            diss = dis.split(' ')
            tokens = sentence.split(' ')
            s1_tokens = s1_sdp.split(' ')
            s2_tokens = s2_sdp.split(' ')

            re_tokens = rewrite_sent.split(" , ")
            s_ws = s_w.split(", ")
            arounds = [word_c for word_c in c_around.split(' ') if word_c != ""] + [word_d for word_d in d_around.split(' ') if word_d != ""]

            pos_word = POS_tag.split(', ')
            pos_string_all = list()
            for i in range(0,len(pos_word),2) :
                pos_string = pos_word[i+1].replace("u'",'').replace("'",'').replace('(','').replace(')','')
                pos_string_all.append(pos_string)
            
            ent_dp = ent_dp.split('] [')
            ent_dp_all = list()
            for ind in range(len(ent_dp)) :
                ent_dp_list = ent_dp[ind].replace('[','').replace(']','').replace("u'",'').replace("'",'').replace('(','').replace(')','').split(', ')

                for i in range(0,len(ent_dp_list),3) :
                    dp = ent_dp_list[i]+'/'+ent_dp_list[i+1]+'/'+ent_dp_list[i+2]
                    ent_dp_all.append(dp)



            labelsDistribution[label] += 1
            chem_val = np.zeros(maxSentenceLen)
            dis_val = np.zeros(maxSentenceLen)
            entity_pair = np.zeros(maxSentenceLen)
            tokenIds = np.zeros(maxSentenceLen)
            
            positionValues1 = np.zeros(maxSentenceLen)
            positionValues2 = np.zeros(maxSentenceLen)
            POS_tokensId = np.zeros(maxSentenceLen)
            close_verb_id = np.zeros(maxSentenceLen)
            retokenIds = np.zeros([6])
            similarIds = np.zeros([6])
            entity_dp_id = np.zeros([6])
            around_id = np.zeros([30]) 
            s1_sdp_id = np.zeros([15])
            s2_sdp_id = np.zeros([15])
            simi_sent1_id = np.zeros([1])
            simi_sent11_id = np.zeros([1])
            simi_sent12_id = np.zeros([1])
            simi_sent2_id = np.zeros([1])
            simi_sent21_id = np.zeros([1])
            simi_sent22_id = np.zeros([1])
           
            for idx in xrange(0, min(maxSentenceLen, len(tokens))):
                
                tokenIds[idx] = getBioWordIdx(tokens[idx], Bioword2Idx)
                
                if len(chems) == 1 :
                    distance1 = idx - int(chem_pos1) # orignal for entity position that it only one word
                if len(diss) == 1 :
                    distance2 = idx - int(dis_pos2) # orignal for entity position that it only one word
                if len(chems) > 1 :
                    if idx < int(chem_pos1) :
                        distance1 = idx - int(chem_pos1)
                    if idx >= int(chem_pos1) and idx <= int(chem_pos1) + len(chems)-1:
                        distance1 = 0 
                    if idx > int(chem_pos1) + len(chems)-1 :
                        distance1 = idx - int(chem_pos1) - len(chems) +1

                if len(diss) > 1 :
                    if idx < int(dis_pos2) :
                        distance2 = idx - int(dis_pos2)
                    if idx >= int(dis_pos2) and idx <= int(dis_pos2) + len(diss)-1:
                        distance2 = 0 
                    if idx > int(dis_pos2) + len(diss)-1 :
                        distance2 = idx - int(dis_pos2) - len(diss) +1        



                
                if distance1 in distanceMapping:
                    positionValues1[idx] = distanceMapping[distance1]

                elif distance1 <= minDistance:
                    positionValues1[idx] = distanceMapping['LowerMin']
                else:
                    positionValues1[idx] = distanceMapping['GreaterMax']
                    
                if distance2 in distanceMapping:
                    positionValues2[idx] = distanceMapping[distance2]

                elif distance2 <= minDistance:
                    positionValues2[idx] = distanceMapping['LowerMin']
                else:
                    positionValues2[idx] = distanceMapping['GreaterMax']            
            
            # version for add coref

            for idx in xrange(0, min(len(re_tokens),6)) :
                if coref_y_n == '1' :
                    retokenIds[0] = 1
                    if re_tokens[0] == 'null' :
                        retokenIds[idx+1] = 0 
                    else :
                        if idx < 5 :
                            retokenIds[idx+1] = 1 

                else :
                    if re_tokens[0] == 'null' :
                        retokenIds[idx] = 0
                    else :
                        retokenIds[idx] = 1

            for idx in xrange(0, min(len(s_ws),6),2) :
               
                if s_ws[0] == 'null null' :
                    similarIds[idx] = 0
                    
                else :
                    
                    similarIds[idx] = getBioWordIdx(s_ws[idx/2].split(' ')[0], Bioword2Idx)
                    similarIds[idx+1] = getBioWordIdx(s_ws[idx/2].split(' ')[1], Bioword2Idx)
                       


            for idx in xrange(0,len(chems)) :
                chem_val[int(chem_pos1)+idx] = getBioWordIdx(chems[idx], Bioword2Idx)
                
            for idx in xrange(0,len(diss)) :
                dis_val[int(dis_pos2)+idx] = getBioWordIdx(diss[idx], Bioword2Idx)
                
            for idx in xrange(0, min(maxSentenceLen,len(pos_string_all))) :
                POS_tokensId[idx] = getBioWordIdx(pos_string_all[idx], Bioword2Idx)
                
            for idx in xrange(0, min(len(ent_dp_all),5)) :
                if ent_dp_all[idx] == 'null/null/null' :
                    entity_dp_id[idx] = -1
                else :
                    entity_dp_id[idx] = 1

            for idx in xrange(0, min(len(arounds),30)) :
                around_id[idx] = getBioWordIdx(arounds[idx], Bioword2Idx)

            for idx in xrange(0, min(len(s1_tokens),15)) :
                s1_sdp_id[idx] = getBioWordIdx(s1_tokens[idx], Bioword2Idx)

            for idx in xrange(0, min(len(s2_tokens),15)) :
                s2_sdp_id[idx] = getBioWordIdx(s2_tokens[idx], Bioword2Idx)

            simi_sent1_id[0] = getSentIdx(sim_sent_1, sentenceIndex)
            simi_sent11_id[0] = getSentIdx(sim_sent_11, sentenceIndex)
            simi_sent12_id[0] = getSentIdx(sim_sent_12, sentenceIndex) 
            simi_sent2_id[0] = getSentIdx(sim_sent_2, sentenceIndex)
            simi_sent21_id[0] = getSentIdx(sim_sent_21, sentenceIndex)
            simi_sent22_id[0] = getSentIdx(sim_sent_22, sentenceIndex)


            chemMatrix.append(chem_val)
            disMatrix.append(dis_val)
            tokenMatrix.append(tokenIds)
            positionMatrix1.append(positionValues1)
            positionMatrix2.append(positionValues2)
            POSMatrix.append(POS_tokensId)
            retokenMatrix.append(retokenIds)
            ent_dpM.append(entity_dp_id)
            s_wMatrix.append(similarIds)
            around_wordM.append(around_id)
            s1_sdpM.append(s1_sdp_id)
            s2_sdpM.append(s2_sdp_id)
            Similar_sent1M.append(simi_sent1_id)
            Similar_sent11M.append(simi_sent11_id)
            Similar_sent12M.append(simi_sent12_id)
            Similar_sent2M.append(simi_sent2_id)
            Similar_sent21M.append(simi_sent21_id)
            Similar_sent22M.append(simi_sent22_id)

            labels.append(labelsMapping[label])
    
    return np.array(labels, dtype='int32'), np.array(tokenMatrix, dtype='int32'), np.array(chemMatrix, dtype='int32'),\
     np.array(disMatrix, dtype='int32'), np.array(positionMatrix1, dtype='int32'), np.array(positionMatrix2, dtype='int32'),\
     np.array(POSMatrix, dtype='int32'), np.array(retokenMatrix, dtype='int32') , np.array(ent_dpM, dtype='int32'), \
     np.array(s_wMatrix, dtype='int32'), np.array(around_wordM, dtype='int32'), np.array(s1_sdpM, dtype='int32'), np.array(s2_sdpM, dtype='int32'), np.array(Similar_sent1M, dtype='int32'),\
     np.array(Similar_sent11M, dtype='int32'),np.array(Similar_sent12M, dtype='int32'), np.array(Similar_sent2M, dtype='int32'), \
     np.array(Similar_sent21M, dtype='int32'), np.array(Similar_sent22M, dtype='int32')
        
        
 
def getWordIdx(token, word2Idx): 
    """Returns from the word2Idex table the word index for a given token"""

    if token in word2Idx:
        return word2Idx[token]
    elif token.lower() in word2Idx:
        return word2Idx[token.lower()]
    
    return word2Idx["UNKNOWN"]

def getBioWordIdx(token, Bioword2Idx): 
    """Returns from the Bioword2Idx table the word index for a given token"""

    if token in Bioword2Idx:
        return Bioword2Idx[token]
    elif token.lower() in Bioword2Idx:
        return Bioword2Idx[token.lower()]
    
    return Bioword2Idx["UNKNOWN"]

def getSentIdx(sent, sentenceIndex) :

    if sent in sentenceIndex:
        return sentenceIndex[sent]
    else :
        print "Unknown sentence :", sent
        return sentenceIndex["None"]
        
    
f_sentence_emb = open(sentence_embPath, 'rb')
sentence_emb = pkl.load(f_sentence_emb)
print len(sentence_emb)
f_sentence_emb.close()


for fileIdx in xrange(len(files)):
    file = files[fileIdx]
    with codecs.open(file, encoding="utf-8") as f1:
        inst_lines = f1.readlines()
        for i in range(len(inst_lines)):

            parts = inst_lines[i].strip('\n').split("\t")
            label = parts[0]
            sentence = parts[9].strip()


            sim_sent_1 = parts[19].strip()
            sim_sent_11 = parts[20].strip()
            sim_sent_12 = parts[21].strip()
            sim_sent_2 = parts[22].strip()
            sim_sent_21 = parts[23].strip()
            sim_sent_22 = parts[24].strip()
            tokens = sentence.split(" ")
            maxSentenceLen[fileIdx] = max(maxSentenceLen[fileIdx], len(tokens))

            if sim_sent_1 in sentence_emb.keys() :
                sents[sim_sent_1] = True

            if sim_sent_11 in sentence_emb.keys() :
                sents[sim_sent_11] = True

            if sim_sent_12 in sentence_emb.keys() :
                sents[sim_sent_12] = True

            if sim_sent_2 in sentence_emb.keys() :
                sents[sim_sent_2] = True

            if sim_sent_21 in sentence_emb.keys() :
                sents[sim_sent_21] = True

            if sim_sent_22 in sentence_emb.keys() :
                sents[sim_sent_22] = True

            for token in tokens:
                words[token.lower()] = True

print("words", len(words))
print("sents", len(sents))
print "Max Sentence Lengths: ", maxSentenceLen

Bioword2Idx = dict()
Bio_embeddings = list()

model = KeyedVectors.load_word2vec_format('/home/hsu/Documents/simple_cnn_cdr/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin', binary=True)

bio_word_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()] # k = word, model.wv[k] = word's vector

if len(Bioword2Idx) == 0: #Add padding+unknown
    Bioword2Idx["_PAD"] = len(Bioword2Idx)
    vector = np.zeros(200) #Zero vector vor 'PADDING' word
    Bio_embeddings.append(vector)

    Bioword2Idx["UNKNOWN"] = len(Bioword2Idx)
    vector = np.random.uniform(-0.5, 0.5, 200)
    Bio_embeddings.append(vector)

for i in range(len(bio_word_list)):
    word = bio_word_list[i][0]
    biovec = bio_word_list[i][1]
    if word in words:   
        Bioword2Idx[word] = len(Bioword2Idx)
        Bio_embeddings.append(biovec)

Bio_embeddings = np.array(Bio_embeddings) 

print "Bio_embeddings", Bio_embeddings.shape

sentenceIndex = {}
sent_emb = list()
f_sentence_emb = open(sentence_embPath, 'rb')
sentence_emb = pkl.load(f_sentence_emb)
f_sentence_emb.close()

if len(sentenceIndex) == 0: 
    sentenceIndex["None"] = len(sentenceIndex)
    sent_vector = np.zeros([1,700]) 
    sent_emb.append(sent_vector)

for sent in sentence_emb.keys() :
    
    if sent in sents :
        sentence_vec = sentence_emb[sent]
        sentenceIndex[sent] = len(sentenceIndex)

        sent_emb.append(sentence_vec)

sent_emb = np.array(sent_emb) 

print "sentence embedding : ", sent_emb.shape
print "Len words: ", len(words)

f = gzip.open(embeddingsPklPath, 'wb')
pkl.dump(Bio_embeddings, f, -1)
f.close()

f = gzip.open(sentence_embPklPath, 'wb')
pkl.dump(sent_emb, f, -1)
f.close()

train_set = createMatrices(files[0], Bioword2Idx, sentenceIndex, max(maxSentenceLen))
test_set = createMatrices(files[1], Bioword2Idx, sentenceIndex, max(maxSentenceLen))


f = gzip.open(outputFilePath, 'wb')
pkl.dump(train_set, f, -1)
pkl.dump(test_set, f, -1)
f.close()

print "Data stored in pkl folder"

for label, freq in labelsDistribution.most_common(100):
    print "%s : %f%%" % (label, 100*freq / float(labelsDistribution.N()))
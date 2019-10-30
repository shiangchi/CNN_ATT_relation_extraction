<<<<<<< HEAD
### CDR轉檔 ：
#### STEP 1 :
>轉換CDR corpus 所提供的檔案 -> 執行 corpus/two_sent_v3.py
轉換後的檔會存在corpus/mult_feat/<br />
舉例 ：     corpus/mult_feat/CDR_TestSet.PubTator_new_4_v7<br />
     corpus/mult_feat/CDR_TrainingSet.PubTator_new_4_v7
#### STEP 2 :
>將存在corpus/mult_feat/底下 的檔案刪除掉已重複在single sentence的relation（single sentence的relation 存在files/ 並已.instances為副檔名）-> 執行 del_unused.py
並一樣存在corpus/mult_feat/<br />
舉例 ： corpus/mult_feat/CDR_TestSet.PubTator_new_4_v7_del_same_7

#### STEP 3 :
>接下來將corpus/mult_feat/CDR_TestSet.PubTator_new_4_v7_del_same_7等檔案轉換成 CNN可讀 -> 執行 CNN/preprocess_CDR_cross.py

### CNN訓練：
>最後執行 -> CNN/CNN.py
得到 test data的預測結果存在 pred_result/ ,
pred_result/cnn_across_only_across (label)
pred_result/cnn_across_prob_only_across (probability)

>轉換成BC5可讀的格式去評估結果
轉換 pred_result/cnn_across_only_across 回 BC5的評估結果可讀的格式
執行 -> CNN/convertopubtator_CNN_CDR.py
儲存於 convert2pub/<br />
舉例 ： convert2pub/cnn_across_only_across

### 最後評估方式 ：
>進入BC5CDR_Evaluation後 執行 ->
sh eval_relation.sh PubTator data/gold/CDR_TestSet.PubTator.txt /Absolute/path/CNN_Att/convert2pub/cnn_across_only_across  eval/cnn_across_only
結果會存在BC5CDR_Evaluation/eval/cnn_across_only
=======
CNN_ATT_relation_extraction
hello
>>>>>>> master

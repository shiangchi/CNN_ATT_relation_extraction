FORMAT=$1
GOLD_FILE=$2
PREDICTION_FILE=$3
SAVE_FILE=$4
# java -cp bc5cdr_eval.jar ncbi.bc5cdr_eval.Evaluate relation CID $FORMAT $GOLD_FILE $PREDICTION_FILE | grep -v INFO # only result, no tp fp...
java -cp bc5cdr_eval.jar ncbi.bc5cdr_eval.Evaluate relation CID $FORMAT $GOLD_FILE $PREDICTION_FILE >> $SAVE_FILE 
# java -cp bc5cdr_eval.jar ncbi.bc5cdr_eval.Evaluate relation CID $FORMAT $GOLD_FILE $PREDICTION_FILE


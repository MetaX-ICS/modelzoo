
OUTPUT=$1
LOGGING=$2

cd ./demo/eval/ic15
python script.py -g=../../../data/gt.zip -s="${OUTPUT}"
cd ../../..
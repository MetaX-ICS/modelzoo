export MXSDK_LOG_LEVEL=1
export MACA_QUANTIZER_USING_CUDA=0
export PYTHONDONTWRITEBYTECODE=1

MISSION_PATH=$1
MODEL_PATH=$2
PRECISION=$3

cd ${MISSION_PATH}
python code/start.py ${MODEL_PATH} 1 ${PRECISION} normal ${MODEL_PATH}
cd ../
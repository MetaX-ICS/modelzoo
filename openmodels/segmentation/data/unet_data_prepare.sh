
cd data
if [ ! -f ~/.kaggle/kaggle.json ];then
    echo -n "kaggle username: "
    read USERNAME
    echo
    echo -n "Kaggle API key: "
    read APIKEY
    
    mkdir -p ~/.kaggle
    echo "{\"username\":\"$USERNAME\",\"key\":\"$APIKEY\"}" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
  
fi

pip install kaggle --upgrade

kaggle competitions download -c carvana-image-masking-challenge -f train_hq.zip
unzip train_hq.zip
mv train_hq/ imgs/


kaggle competitions download -c carvana-image-masking-challenge -f train_masks.zip
unzip train_masks.zip
mv train_masks/ masks/
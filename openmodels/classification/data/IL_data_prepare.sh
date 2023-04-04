cd data
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
mkdir val_old/
mv ILSVRC2012_img_val.tar val_old/
cd val_old
tar xvf ILSVRC2012_img_val.tar
cd ../
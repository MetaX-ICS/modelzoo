
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
mv ILSVRC2012_img_val.tar data/
cd data
mkdir val/
tar xvf ILSVRC2012_img_val.tar
mv *.JPEG val/
cd ../
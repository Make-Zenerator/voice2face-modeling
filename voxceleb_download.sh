apt-get install fastjar \
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa \
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab \
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac \
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad \
cat vox1_dev* > vox1_dev_wav.zip \
rm -rf vox1_dev_wav_part* \
jar xvf vox1_dev_wav.zip \
mv wav dev \
mkdir data/VoxCeleb/raw_wav/vox1 \
mv dev data/VoxCeleb/raw_wav/vox1/dev \

wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv \
mv vox1_meta.csv data/VoxCeleb/vox1 \

wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip \
unzip vox1_test_wav.zip \
rm -rf vo1_test_wav.zip \
mv wav test \
mv test data/VoxCeleb/raw_wav/vox1/test
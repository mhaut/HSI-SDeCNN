
#!/bin/bash

if [ -d "data" ]; then
    read -p "Do you wish to overwrite data folder (CAUTION: ACTUAL FOLDER WILL BE DELETED)?" yn
    case $yn in
        [Yy]* ) rm -r data;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no."; exit;;
    esac
fi

mkdir data


echo 'Retrieving Indian Pines...'
# In the repo

echo 'Retrieving University of Pavia...'
wget 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat' -O data/PaviaU.mat
wget 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat' -O data/PaviaU_gt.mat

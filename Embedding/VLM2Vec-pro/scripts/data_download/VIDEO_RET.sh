#!/bin/bash
NC='\033[0m'
print_red() {
    RED='\033[0;31m'
    echo -e "${RED}$1${NC}"
}
print_yellow() {
    YELLOW='\033[1;33m'
    echo -e "${YELLOW}$1${NC}"
}
print_green() {
    GREEN='\033[0;32m'
    echo -e "${GREEN}$1${NC}"
}

print_red "This script is used to retrieve raw videos of the train split of MMEB V2 CLS dataset."
print_red "It will create 4 folders in the current directory or the one specified, so please make sure it is executed in a desired place."
print_red "It is recommended to run this script in a tmux session, as it may take a long time \(more than 10 hours\) to complete."

root_dir=$(pwd)
if [ $# -eq 1 ]; then
    root_dir="$1"
    [ ! -d "$root_dir" ] && mkdir -p "$root_dir"
    echo "The root directory is set to $root_dir"
else
    echo "No root directory specified, using the current directory $root_dir"
fi

print_yellow "Downloading Something Something V2 ... ... [1/4]"
ssv2_dir="${root_dir}/SSv2"
[ ! -d "$ssv2_dir" ] && mkdir -p "$ssv2_dir/videos"
cd $ssv2_dir
wget -c https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-00
wget -c https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-01
echo Extracting Something Something V2 dataset ... ...
cat 20bn-something-something-v2-?? | tar -xvz --strip-components=1 -C "$ssv2_dir/videos"
echo Extracting ssv2 complete, removing the tar files
rm 20bn-something-something-v2-??
print_green "Downloading and extracting SSv2 complete!"
cd $root_dir

print_yellow "Downloading UCF101 ... ... [2/4]"
ucf101_dir="${root_dir}/UCF101"
[ ! -d "$ucf101_dir" ] && mkdir -p "$ucf101_dir"
cd $ucf101_dir
wget -c --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
echo Extracting UCF101 dataset ... ...
unrar x UCF101.rar .
echo Extracting ucf101 complete, removing the rar files ... ...
rm UCF101.rar
print_green "Downloading and extracting UCF101 complete!"
cd $root_dir

print_yellow "Downloading HMDB51 ... ... [3/4]"
hmdb51_dir="${root_dir}/HMDB51"
[ ! -d "$hmdb51_dir" ] && mkdir -p "$hmdb51_dir/videos_sta"
cd $hmdb51_dir
wget -c http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_sta.rar
echo Extracting HMDB51 dataset ... ...
unrar x hmdb51_sta.rar videos_sta/
cd videos_sta/
find . -name "*.rar" -exec unrar x {} \;
echo Extracting hmdb51 complete, removing the rar files ... ...
rm *.rar ../*.rar
print_green "Downloading and extracting HMDB51 complete!"
cd $root_dir

print_yellow "Downloading Kinetics 700 2020 ... ... [4/4]"
k700_dir="${root_dir}/K700"
k700_dir_targz="${k700_dir}/videos_targz"
[ ! -d "$k700_dir" ] && mkdir -p $k700_dir_targz
wget -c -i https://s3.amazonaws.com/kinetics/700_2020/train/k700_2020_train_path.txt -P $k700_dir_targz
echo Extracting Kinetics 700 2020 dataset ... ...
k700_dir_videos="${k700_dir}/videos"
[ ! -d "$k700_dir_videos" ] && mkdir -p $k700_dir_videos
tar_list=$(ls $k700_dir_targz)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $k700_dir_targz/$f to $k700_dir_videos && tar zxf $k700_dir_targz/$f -C $k700_dir_videos
done
echo Extracting k700 complete, removing the tar files ... ...
rm $k700_dir_targz/*.tar.gz
rm -r $k700_dir_targz
cd $root_dir
print_green "Downloading and extracting K700 complete!"

print_red "All datasets have been downloaded and extracted successfully!"
print_red "Please check the folders: There should be 4 folders in the current directory, each containing a folder called 'videos' (except for HMDB51 which is videos_sta)."
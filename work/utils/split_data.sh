set -e
set -x

SRC_LNG=hi
TGT_LNG=mr

#train:
sed -n '3001,$p' train.${SRC_LNG}-${TGT_LNG}.${SRC_LNG}.dedup.shuf > train.${SRC_LNG}-${TGT_LNG}.${SRC_LNG}
sed -n '3001,$p' train.${SRC_LNG}-${TGT_LNG}.${TGT_LNG}.dedup.shuf > train.${SRC_LNG}-${TGT_LNG}.${TGT_LNG}

#val:
sed -n '2001,3000p' train.${SRC_LNG}-${TGT_LNG}.${SRC_LNG}.dedup.shuf > val.${SRC_LNG}-${TGT_LNG}.${SRC_LNG}
sed -n '2001,3000p' train.${SRC_LNG}-${TGT_LNG}.${TGT_LNG}.dedup.shuf > val.${SRC_LNG}-${TGT_LNG}.${TGT_LNG}

#test:
sed -n '1,2000p' train.${SRC_LNG}-${TGT_LNG}.${SRC_LNG}.dedup.shuf > test.${SRC_LNG}-${TGT_LNG}.${SRC_LNG}
sed -n '1,2000p' train.${SRC_LNG}-${TGT_LNG}.${TGT_LNG}.dedup.shuf > test.${SRC_LNG}-${TGT_LNG}.${TGT_LNG}


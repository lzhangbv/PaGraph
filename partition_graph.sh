partition=1
preprocess=1
hops=2

if [ "$preprocess" = "1" ]; then
hops=$(expr $hops - 1)
fi

echo "--num-partitions $partition --preprocess $preprocess --num-hops $hops"

# hash partition
#python PaGraph/partition/hash.py --num-hops $hops --partition $partition --dataset /localdata/reddit

# dg partition
python PaGraph/partition/dg.py --num-hops $hops --partition $partition --dataset /localdata/reddit

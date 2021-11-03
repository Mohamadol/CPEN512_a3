rm -rf results
mkdir ./results

for X in 1 8 16 128 512 1024 2048 4096
do
    mkdir ./results/threads_in_block_$X
    for Y in 64 128 256 512 1024 2048 4096
    do
        echo "threads_in_block  $X   N  $Y"
        mkdir ./results/threads_in_block_$X/N$Y
        nvcc source/a3.cu -o a3 -O3 -m64 -D THREADS_IN_BLOCK=$X -D N=$Y
        ./a3 > ./results/threads_in_block_$X/N$Y/result.txt
    done
done

alpha=0.0
numstep=500
use_fid_inception=True
lightning=True
mixoption="True False"
trainstep=200000
T0=1.0
batch=64

python3 cifar_precompute.py 32 0 True

option="default noise data"

for mixed in $mixoption; do
for weight in $option; do
	for seed in {0..1}; do
        # this seed is the seed for training, not for generating samples in computing FID

		echo "$weight $seed"
		python3 cifar_fid.py --alpha=$alpha --seed=0 --device_id=0 --num_sample=10500 --fid_batch_size=750 --num_steps=$numstep \
            --use_fid_inception=$use_fid_inception \
			--ckpt_path="./saved/cifar-$weight-$batch-$trainstep-$T0-$seed-$lightning-$mixed/checkpoint-iter-40000.pt" &
		python3 cifar_fid.py --alpha=$alpha --seed=0 --device_id=1 --num_sample=10500 --fid_batch_size=750 --num_steps=$numstep \
            --use_fid_inception=$use_fid_inception \
			--ckpt_path="./saved/cifar-$weight-$batch-$trainstep-$T0-$seed-$lightning-$mixed/checkpoint-iter-80000.pt" &
		python3 cifar_fid.py --alpha=$alpha --seed=0 --device_id=2 --num_sample=10500 --fid_batch_size=750 --num_steps=$numstep \
            --use_fid_inception=$use_fid_inception \
			--ckpt_path="./saved/cifar-$weight-$batch-$trainstep-$T0-$seed-$lightning-$mixed/checkpoint-iter-120000.pt" 

		python3 cifar_wait.py

		python3 cifar_fid.py --alpha=$alpha --seed=0 --device_id=0 --num_sample=10500 --fid_batch_size=750 --num_steps=$numstep \
            --use_fid_inception=$use_fid_inception \
			--ckpt_path="./saved/cifar-$weight-$batch-$trainstep-$T0-$seed-$lightning-$mixed/checkpoint-iter-160000.pt" &
		python3 cifar_fid.py --alpha=$alpha --seed=0 --device_id=1 --num_sample=10500 --fid_batch_size=750 --num_steps=$numstep \
            --use_fid_inception=$use_fid_inception \
			--ckpt_path="./saved/cifar-$weight-$batch-$trainstep-$T0-$seed-$lightning-$mixed/checkpoint.pt"

		python3 cifar_wait.py

	done
done
done

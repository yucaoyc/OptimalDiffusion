lightning=True
trainstep=200000
T0=1.0
batch=64
Nt=50
option="default noise data"
mixoption="True False"
device0=0
device1=1
device2=2

for mixed in $mixoption; do
for weight in $option; do
	for seed in {0..1}; do
        # this seed is the seed for training, not for generating samples in computing SML

		echo "$weight $seed"
        python3 cifar_sml.py --nt=$Nt --device_id=$device0 \
			--ckpt_path="./saved/cifar-$weight-$batch-$trainstep-$T0-$seed-$lightning-$mixed/checkpoint-iter-40000.pt" &
        python3 cifar_sml.py --nt=$Nt --device_id=$device1 \
			--ckpt_path="./saved/cifar-$weight-$batch-$trainstep-$T0-$seed-$lightning-$mixed/checkpoint-iter-80000.pt" &
        python3 cifar_sml.py --nt=$Nt --device_id=$device2 \
			--ckpt_path="./saved/cifar-$weight-$batch-$trainstep-$T0-$seed-$lightning-$mixed/checkpoint-iter-120000.pt" 

		python3 cifar_wait.py

        python3 cifar_sml.py --nt=$Nt --device_id=$device0 \
			--ckpt_path="./saved/cifar-$weight-$batch-$trainstep-$T0-$seed-$lightning-$mixed/checkpoint-iter-160000.pt" &
        python3 cifar_sml.py --nt=$Nt --device_id=$device1 \
			--ckpt_path="./saved/cifar-$weight-$batch-$trainstep-$T0-$seed-$lightning-$mixed/checkpoint.pt"

		python3 cifar_wait.py

	done
done
done

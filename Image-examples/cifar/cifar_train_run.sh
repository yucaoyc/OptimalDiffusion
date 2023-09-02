numiter=200000
#mixed=True
mixed=False

python3 cifar_train_model.py --weight_type='default' --seed=0 \
	--num_iterations=$numiter --num_method='ei' \
	--to_plot=True \
	--fabric_device_id=0 \
	--use_mixed_precision=$mixed --save_ckpt_every_iter=True &
python3 cifar_train_model.py --weight_type='default' --seed=1 \
	--num_iterations=$numiter --num_method='ei' \
	--to_plot=True \
	--fabric_device_id=1 \
	--use_mixed_precision=$mixed --save_ckpt_every_iter=True &
python3 cifar_train_model.py --weight_type='data' --seed=0 \
	--num_iterations=$numiter --num_method='ei' \
	--to_plot=True \
	--fabric_device_id=2 \
	--use_mixed_precision=$mixed --save_ckpt_every_iter=True

python3 cifar_wait.py;

python3 cifar_train_model.py --weight_type='noise' --seed=0 \
	--num_iterations=$numiter --num_method='ei' \
	--to_plot=True \
	--fabric_device_id=0 \
	--use_mixed_precision=$mixed --save_ckpt_every_iter=True & 
python3 cifar_train_model.py --weight_type='noise' --seed=1 \
	--num_iterations=$numiter --num_method='ei' \
	--to_plot=True \
	--fabric_device_id=1 \
	--use_mixed_precision=$mixed --save_ckpt_every_iter=True &
python3 cifar_train_model.py --weight_type='data' --seed=1 \
	--num_iterations=$numiter --num_method='ei' \
	--to_plot=True \
	--fabric_device_id=2 \
	--use_mixed_precision=$mixed --save_ckpt_every_iter=True

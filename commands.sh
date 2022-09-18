python main2.py --cfg configs/GPS/default-test-config.yaml wandb.use False --gpu_dev 0

python main2.py --cfg configs/GPS/gps-vsmall-heads.yaml wandb.use False --gpu_dev 1
python main2.py --cfg configs/GPS/gps-vsmall-hiddendim.yaml wandb.use False --gpu_dev 2
python main2.py --cfg configs/GPS/gps-vsmall-layers.yaml wandb.use False --gpu_dev 3

python main2.py --cfg configs/GPS/gps-vvsmall-heads.yaml wandb.use False --gpu_dev 1
python main2.py --cfg configs/GPS/gps-vvsmall-hiddendim.yaml wandb.use False --gpu_dev 2
python main2.py --cfg configs/GPS/gps-vvsmall-layers.yaml wandb.use False --gpu_dev 3
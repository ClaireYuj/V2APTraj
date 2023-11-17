
# Environment
```
pip install -r requirements.txt
```

## Train
The Default settings are to train on ETH-univ. Data cache and models will be in the subdirectory "./savedata/0/".

```
cd GATraj
python train.py --test_set <dataset to train> --num_epochs 1000 --x_encoder_layers 3 --eta_min 1e-5  --batch_size 32\
  --learning_rate 5e-4  --randomRotate True --final_mode 20 --neighbor_thred 10\
  --using_cuda True --clip 1 --pass_time 2 --ifGaussian False --SR True --input_offset True 
```

Configuration files are also created after the first run, arguments could be modified through configuration files or command line. 
Priority: command line \> configuration files \> default values in script.


The datasets are selected on arguments '--test_set'. Five datasets in ETH/UCY are corresponding to the value of \[0,1,2,3,4\] ([**eth, hotel, zara1, zara2, univ**]). 

### Example

This command is to train model for ETH-univ
```
python train.py --test_set 0 --num_epochs 1000 --x_encoder_layers 3 --eta_min 1e-5  --batch_size 32\
  --learning_rate 5e-4  --randomRotate True --final_mode 20 --neighbor_thred 10\
  --using_cuda True --clip 1 --pass_time 2 --ifGaussian False --SR True --input_offset True
  
```
数据集的第一行为帧数（视频所在的帧数），第二行为id（行人id），第三行（y坐标）和第四行（x坐标）为坐标
current_x = FrameContainPed[3, FrameContainPed[0, :] == frame][0]
current_y = FrameContainPed[2, FrameContainPed[0, :] == frame][0]
## Test
We provide the trained model weights in the subdirectory "./savedata/".
This command is to test model for ETH-univ, just add --phase test --load_model 1000 to the end of this training command.
```
python train.py --test_set 0 --num_epochs 1000 --x_encoder_layers 3 --eta_min 1e-5  --batch_size 32\
  --learning_rate 5e-4  --randomRotate True --final_mode 20 --neighbor_thred 10\
  --using_cuda True --clip 1 --pass_time 2 --ifGaussian False --SR True --input_offset True --phase test  --load_model 1000
```

### Reference

The code base heavily borrows from [SR-LSTM](https://github.com/zhangpur/SR-LSTM). The visulization code base for nuScenes is adopted from [PGP](https://github.com/nachiket92/PGP).

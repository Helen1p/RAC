"""train TransRAC model """
from platform import node
import os
## if your data is .mp4 form, please use RepCountA_raw_Loader.py (slowly)
# from dataset.RepCountA_raw_Loader import MyData
## if your data is .npz form, please use RepCountA_Loader.py. It can speed up the training
from dataset.RepCountA_Loader import MyData
# you can use 'tools.video2npz.py' to transform .mp4 to .npz
from models.TransRAC import TransferModel
from training.train_looping import train_loop

# CUDA environment
N_GPU = 1
device_ids = [i for i in range(N_GPU)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# # # we pick out the fixed frames from raw video file, and we store them as .npz file
# # # we currently support 64 or 128 frames
# data root path
root_path = '/home/likun/code/RAC/dataset/LLSP/LLSP_npz/'

train_video_dir = 'train'
train_label_dir = 'train.csv'
valid_video_dir = 'valid'
valid_label_dir = 'valid.csv'

# please make sure the pretrained model path is correct
checkpoint = './pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth'
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'

# TransRAC trained model checkpoint, we will upload soon.
lastckpt = None

NUM_FRAME = 64
# multi scales(list). we currently support 1,4,8 scale.
SCALES = [1, 4, 8]

train_dataset = MyData(root_path, train_video_dir, train_label_dir, num_frame=NUM_FRAME)
valid_dataset = MyData(root_path, valid_video_dir, valid_label_dir, num_frame=NUM_FRAME)
my_model = TransferModel(config=config, checkpoint=checkpoint, num_frames=NUM_FRAME, scales=SCALES, OPEN=False)
NUM_EPOCHS = 200
LR = 1e-4
BATCH_SIZE = 32


### 使用torchinfo summary
from torchinfo import summary
input_size = (1, 3, 64, 244, 244)
model_info = summary(my_model, input_size=input_size, verbose=0)
print(f"GFLOPs: {model_info.total_mult_adds / 1e9}") # 146.55G  total parameters 42,284,743 = 42.28M

#############
# # 计算总参数量
# 计算总参数量，并转换为兆（M）
total_params = sum(p.numel() for p in my_model.parameters()) / 1e6
# 计算需要梯度的参数量，并转换为兆（M）
trainable_params = sum(p.numel() for p in my_model.parameters() if p.requires_grad) / 1e6
print(f"Total parameters: {total_params:.2f}M")
print(f"Trainable parameters: {trainable_params:.2f}M") # 42.28M

# train_loop(NUM_EPOCHS, my_model, train_dataset, valid_dataset, train=True, valid=True,
#            batch_size=BATCH_SIZE, lr=LR, saveckpt=True, ckpt_name='ours', log_dir='ours', device_ids=device_ids,
#            lastckpt=lastckpt, mae_error=False)

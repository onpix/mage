import json
import torch
import random
random.seed(1433)

pcd_feats_norgb_46k_path='/mnt/hwfile/3dobject_aigc/hezexin/datasets/objaverse-pcd-ours-feats-norgb/caches/0620_42k_partial/5.pt'


pcd_feats_no_rgb=torch.load(pcd_feats_norgb_46k_path)
uids_with_pcd_feats_filtered=[]
# filter pcd feats with scores > 0.65
socre_threshold=0.7
filterd_inds=torch.where(pcd_feats_no_rgb['scores'].max(dim=1)[0]>socre_threshold)[0]

# lvis uids
# lvis_uids=torch.load('/mnt/hwfile/3dobject_aigc/hezexin/datasets/objaverse-pcd-lvis-feats/top_k_cache/3.pt')['uids']
for i in filterd_inds:
    # if i not in lvis_uids:
    uids_with_pcd_feats_filtered.append(pcd_feats_no_rgb['uids'][i])


# split train val test set
# sample 32 uids for validation and 1000 uids for test
random.shuffle(uids_with_pcd_feats_filtered)
    
test_uids=uids_with_pcd_feats_filtered[:1000]
val_uids=test_uids[:32]

assert len(val_uids)==32
assert len(test_uids)==1000

val_uids[:23]=[
    "2f4e512e045c4922b83bc113ad7c1ca4",
    "7dd3ffc97d224de396aff54f1e728386",
    "19fb4444c70942babe4ae80e041e38bf",
    "9eebc5ccc02744fabc87640890a1df4d",
    "317f5731caa146f5b77af0bbc6c942c9",
    "1637cec2eef34eb493f017e2fe78ad8e",
    "e93d3b389ea04c9ca8e532b7471a1cb0",
    "c89f01b2d4bc4a27981d26ddb9b0e49d",
    "b245d60aa599445b85673581e24ded92",
    "b51fbe92f3114f8a89f509924daecc75",
    "b9beb81feeb247c89e66c389f8606d1a",
    "a654dcfdd5744259a88885c20d14cd9b",
    "42818615ef1f49ef909683031c5f5f38",
    "466726edcb7f468f97855195e0f913d7",
    "4862b909db9d45b9b8c652de18815eeb",
    "4189da714f07441c8d8c0f34a408ef78",
    "422fb3960e3e45cb86e6afcd8a8f223f",
    "222ce0d000dd4c75803cdbfed14059a9",
    "99acde3dfaff4c9bb590e986fb35a66b",
    "2eb520b6ddc04ed695206174b2a3fb25",
    "1c6fc0dad15844e3ade5d03411d64ce1",
    "0cdf5b24bf504265b3bc2168cf7885ca",
    "0a843e55cd2f4cd68dffb9056ef4a061",
]
# test_uids=val_uids
# print(len(val_uids))

all_uids_path='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3/data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1/meta.json'
all_uids=json.load(open(all_uids_path))

train_uids=[]
for uid in all_uids:
    if uid not in val_uids and uid not in test_uids:
        train_uids.append(uid)

print(f'train: {len(train_uids)}')
print(f'val: {len(val_uids)}')
print(f'test: {len(test_uids)}')

# save sets as json file
# train_dir='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3/data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1_z123-pcd-feat-norgb/train.json'
# val_dir='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3/data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1_z123-pcd-feat-norgb/val.json'
# test_dir='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3/data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1_z123-pcd-feat-norgb/test.json'
train_dir='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3/data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1_z123-pcd-feat-norgb_1000test_v2/train.json'
val_dir='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3/data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1_z123-pcd-feat-norgb_1000test_v2/val.json'
test_dir='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3/data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1_z123-pcd-feat-norgb_1000test_v2/test.json'

# dump json
json.dump(train_uids, open(train_dir,'w'))
json.dump(val_uids, open(val_dir,'w'))
json.dump(test_uids, open(test_dir,'w'))
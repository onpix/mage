import os
import json

noscan_dir='data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan/meta.json'
noscan_train_dir='data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan/train.json'
noscan_val_dir='data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan/val.json'

noscan_lvis_dir='data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lvis/meta.json'
noscan_lvis_train_dir='data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lvis/train.json'
noscan_lvis_val_dir='data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lvis/val.json'

lgm_v1_dir='data_train/Objaverse/rendered/LGM_training/kiuisobj_v1_merged_80K.csv'

print('noscan_uids')
noscan_uids=json.load(open(noscan_dir,'r'))
print(len(noscan_uids), noscan_uids[:3])

noscan_train_uids=json.load(open(noscan_train_dir,'r'))
print(len(noscan_train_uids), noscan_train_uids[:3])

noscan_val_uids=json.load(open(noscan_val_dir,'r'))
print(len(noscan_val_uids), noscan_val_uids[:3])

print('noscan_lvis_uids')
noscan_lvis_uids=json.load(open(noscan_lvis_dir,'r'))
print(len(noscan_lvis_uids), noscan_lvis_uids[:3])

noscan_lvis_train_uids=json.load(open(noscan_lvis_train_dir,'r'))
print(len(noscan_lvis_train_uids), noscan_lvis_train_uids[:3])

noscan_lvis_val_uids=json.load(open(noscan_lvis_val_dir,'r'))
print(len(noscan_lvis_val_uids), noscan_lvis_val_uids[:3])

print('lgm_v1_uids')

lgm_v1_uids=[]
for line in open(lgm_v1_dir,'r').readlines():
    lgm_v1_uids.append(line.split(',')[-1].strip())

print(len(lgm_v1_uids), lgm_v1_uids[:3])

inter_uids=list(set(noscan_uids).intersection(set(lgm_v1_uids)))
inter_train_uids=list(set(noscan_train_uids).intersection(set(lgm_v1_uids)))
inter_val_uids=list(set(noscan_val_uids).intersection(set(lgm_v1_uids)))
print(f'intersection lgmv1 & noscan: {len(inter_uids)}')
print(f'intersection lgmv1 & noscan_train: {len(inter_train_uids)}')
print(f'intersection lgmv1 & noscan_val: {len(inter_val_uids)}')

# save the intersection uids as meta.json under 'data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1/meta.json'
inter_dir='data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1/meta.json'
json.dump(inter_uids, open(inter_dir,'w'))

inter_train_dir='data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1/train.json'
json.dump(inter_train_uids, open(inter_train_dir,'w'))

inter_val_dir='data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1/val.json'
json.dump(inter_val_uids, open(inter_val_dir,'w'))

# read the intersection uids
inter_uids=json.load(open(inter_dir,'r'))
print(len(inter_uids), inter_uids[:3])

inter_train_uids=json.load(open(inter_train_dir,'r'))
print(len(inter_train_uids), inter_train_uids[:3])

inter_val_uids=json.load(open(inter_val_dir,'r'))
print(len(inter_val_uids), inter_val_uids[:3])


# uid with glb path
models_path_all='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/LRM-Rendering/models_paths/models_path_all.json'
uid2glb=json.load(open(models_path_all,'r'))
inter_uid2glb={}
for uid in inter_uids:
    inter_uid2glb[uid]=uid2glb[uid]
print(len(inter_uid2glb), list(inter_uid2glb.items())[:3])
inter_uid2glb_dir='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/LRM-Rendering/models_paths/models_path_noscan_lgmv1.json'
json.dump(inter_uid2glb, open(inter_uid2glb_dir,'w'))

# read
inter_uid2glb=json.load(open(inter_uid2glb_dir,'r'))
print(len(inter_uid2glb), list(inter_uid2glb.items())[:3])

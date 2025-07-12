import os
import json
import megfile

def my_smart_exists(remote_output_dir, uid):
    if megfile.smart_exists(megfile.smart_path_join(remote_output_dir, uid, "pose")) and \
        megfile.smart_exists(megfile.smart_path_join(remote_output_dir, uid, "rgba")) and \
        megfile.smart_exists(megfile.smart_path_join(remote_output_dir, uid, "ccm")) and \
        megfile.smart_exists(megfile.smart_path_join(remote_output_dir, uid, "normal")):
        len_pose = len(
            megfile.smart_listdir(megfile.smart_path_join(remote_output_dir, uid, "pose"))
        )
        len_rgba = len(
            megfile.smart_listdir(megfile.smart_path_join(remote_output_dir, uid, "rgba"))
        )
        len_ccm = len(
            megfile.smart_listdir(megfile.smart_path_join(remote_output_dir, uid, "ccm"))
        )
        len_normal = len(
            megfile.smart_listdir(megfile.smart_path_join(remote_output_dir, uid, "normal"))
        )

        intrinsics_exist = megfile.smart_exists(
            megfile.smart_path_join(remote_output_dir, uid, "intrinsics.npy")
        )
        res = intrinsics_exist and len_pose == 32 * 2 and len_rgba == 32 and len_ccm == 32 and len_normal == 32
        return res
    else: 
        return False
    


remote_output_dir = "s3+my_s_hdd_new://objaverse-render-lgm/beta"
remote_output_div2_dir = "s3+my_s_hdd://objaverse-render-lgm-div2/beta"

train_uid_path='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3/data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1/train.json'
val_uid_path='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3/data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1/val.json'
split_uid_path='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/LRM-Rendering/models_paths/model_path_noscan_lgmv1_smaller_split/models_path_noscan_lgmv1_12.json'
train_uids=json.load(open(train_uid_path,'r'))
val_uids=json.load(open(val_uid_path,'r'))
split_uids=json.load(open(split_uid_path,'r')).keys()
print(f'train: {len(train_uids)}')
print(f'val: {len(val_uids)}')
print(f'split: {len(split_uids)}')

train_uids_rendered=[]
train_uids_not_rendered=[]
val_uids_rendered=[]
val_uids_not_rendered=[]

count=0
for uid in train_uids:
    # if count>120:
    #     break
    if split_uids is not None:
        if uid not in split_uids:
            continue
    if my_smart_exists(remote_output_div2_dir, uid) or my_smart_exists(remote_output_dir, uid):
        train_uids_rendered.append(uid)
        count+=1
        print(count, "========", uid, "rendered", "========")
    else:
        train_uids_not_rendered.append(uid)
        print("========", uid, "not rendered", "========")

count=0
for uid in val_uids:
    # if count>16:
    #     break
    if split_uids is not None:
        if uid not in split_uids:
            continue
    if my_smart_exists(remote_output_div2_dir, uid) or my_smart_exists(remote_output_dir, uid):
        val_uids_rendered.append(uid)
        count+=1
        print(count, "========", uid, "rendered", "========")
    else:
        val_uids_not_rendered.append(uid)
        print("========", uid, "not rendered", "========")

print(f'train rendered: {len(train_uids_rendered)}')
print(f'train not rendered: {len(train_uids_not_rendered)}')
print(f'val rendered: {len(val_uids_rendered)}')
print(f'val not rendered: {len(val_uids_not_rendered)}')

train_rendered_uid_path='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3/data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1/train_rendered.json'
train_not_rendered_uid_path='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3/data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1/train_not_rendered.json'
val_rendered_uid_path='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3/data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1/val_rendered.json'
val_not_rendered_uid_path='/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3/data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1/val_not_rendered.json'
json.dump(train_uids_rendered, open(train_rendered_uid_path,'w'))
json.dump(train_uids_not_rendered, open(train_not_rendered_uid_path,'w'))
json.dump(val_uids_rendered, open(val_rendered_uid_path,'w'))
json.dump(val_uids_not_rendered, open(val_not_rendered_uid_path,'w'))
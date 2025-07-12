import torch
from megfile import smart_copy, smart_path_join, smart_exists
from megfile.errors import S3UnknownError
import time
import os
import numpy as np

def _locate_datadir(root_dirs, uid, locator: str):
    while True:
        try:
            for root_dir in root_dirs:
                datadir = smart_path_join(root_dir, uid, locator)
                if smart_exists(datadir):
                    return root_dir
            # print(f"Cannot find valid data directory for uid {uid}")
            return None
        #  megfile.errors.S3UnknownError
        except S3UnknownError: 
            print(f'S3UnknownError: {datadir}')
            time.sleep(2)


if __name__ == "__main__":
    rendered_root_dirs=(
        "/mnt/hwfile/3dobject_aigc/wangzhenwei/datasets/objaverse-render-lgm-div2/beta",
        "/mnt/hwfile/3dobject_aigc/wangzhenwei/datasets/objaverse-render-lgm/beta/",
    )
    
    # topk_cache_path='/mnt/hwfile/3dobject_aigc/hezexin/datasets/objaverse-pcd-ours-feats/caches/0619_42k_partial/5.pt'
    topk_cache_path='/mnt/hwfile/3dobject_aigc/hezexin/datasets/objaverse-pcd-ours-feats-norgb/caches/0620_42k_partial/5.pt'
    prefix='42k_lvis_lgm_newpcd_norgb_score0.3-0.5'
    save_path=f'/mnt/hwfile/3dobject_aigc/wangzhenwei/datasets/ref_mv_testset/{prefix}'
    topk_cache = torch.load(topk_cache_path)
    # import pdb; pdb.set_trace()
    uids=np.array(topk_cache['uids'])
    scores_list=np.array(topk_cache['scores'])
    topk_inds_list=np.array(topk_cache['inds'])
    # Shuffle two lists with same order
    np.random.seed(1)
    random_order=np.random.permutation(len(uids))
    print(random_order)
    rand_order_uids=uids[random_order]
    rand_order_scores=scores_list[random_order]
    topk_inds_list=topk_inds_list[random_order]

    count=0
    for inds, uid in enumerate(rand_order_uids):
        # if uid not in ['0ac34c3eb29f40219c554ae8445860c9','07aefdcb6d6e41feb2890373caf83c9d','ab41d89037394fbf9e0bdcf0629eda37']:
            # continue
        print(uid)
        topk_inds=topk_inds_list[inds]
        topk_scores=rand_order_scores[inds]
        for topi_ind,topi_score in zip(topk_inds,topk_scores):
            # if topi_score>0.5:
                # continue
            uid_topi=uids[topi_ind]
            root_dir = _locate_datadir(rendered_root_dirs, uid, 'intrinsics.npy')
            root_dir_topi = _locate_datadir(rendered_root_dirs, uid_topi, 'intrinsics.npy')
            if root_dir is not None and root_dir_topi is not None:
                print(f'find pared objects {uid} and {uid_topi}')
                rgba_dir = os.path.join(root_dir, uid, "rgba")
                rgba_dir_topi = os.path.join(root_dir_topi, uid_topi, "rgba")
                ccm_dir_topi = os.path.join(root_dir_topi, uid_topi, "ccm")

                front_view_path=smart_path_join(rgba_dir, f"000.png")
                ref_rgba_paths=[]
                ref_ccm_paths=[]
                for vid in list(range(8)):
                    ref_rgba_paths.append(smart_path_join(rgba_dir_topi, f"{vid:03d}.png"))
                    ref_ccm_paths.append(smart_path_join(ccm_dir_topi, f"{vid:03d}_0001.png"))
                smart_copy(src_path=front_view_path, dst_path=os.path.join(save_path, "ref_front",f"{uid}_{uid_topi}.png"))
                for i in range(8):
                    print(f'save {ref_rgba_paths[i]} to {os.path.join(save_path, "ref_maps_zerp123",f"{uid}_{uid_topi}", "rgba",f"{i:03d}.png")}')
                    smart_copy(src_path=ref_rgba_paths[i], dst_path=os.path.join(save_path, "ref_maps_zero123",f"{uid}_{str(topi_score)}_{uid_topi}", "rgba",f"{i:03d}.png"))
                    smart_copy(src_path=ref_ccm_paths[i], dst_path=os.path.join(save_path, "ref_maps_zero123", f"{uid}_{str(topi_score)}_{uid_topi}", "ccm",f"{i:03d}_0001.png"))
                count+=1
            if count==50:
                exit()
    #         count+=1
    # print(count)
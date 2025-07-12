import shutil
import os

src='./data_test/ref_maps_zero123'
tgt='./data_test/ref_front'

for obj_name in os.listdir(src):
    if obj_name == '.DS_Store':
        continue
    front_image_path = os.path.join(src,obj_name,'rgba','000.png')
    shutil.copyfile(front_image_path,os.path.join(tgt,obj_name+'.png'))
    print(f'save to {os.path.join(tgt,obj_name+".png")}')
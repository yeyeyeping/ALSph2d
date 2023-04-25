import psutil
import time
import os


def wait_snap_kill():
    while True:
        find = False
        for proc in psutil.process_iter():
            try:
                pinfo = proc.as_dict(attrs=['name'])
            except psutil.NoSuchProcess:
                pass
            else:
                if pinfo["name"] == "ITK-SNAP":
                    find = True
        if not find:
            return
        time.sleep(1)


folder = "/home/yeep/dataset/3d/SPH/"
img_folder = f"{folder}/imagesTr"
label_folder = f"{folder}/labelsTr"
names = sorted(os.listdir(img_folder))
for name in names:
    img = f"{img_folder}/{name}"
    label = f"{label_folder}/{name}"
    print(name)
    os.system(f"/home/yeep/app/itksnap-4.0.0-20230220-Linux-gcc64/bin/itksnap -g {img} -s {label}")
    wait_snap_kill()

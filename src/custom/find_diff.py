import os
import shutil


def find_diff():
    dir_name = 'E:\\scrawl_images\\star_images_160\\'
    org_dirname = 'E:\\scrawl_images\\star_images\\'
    set_names = set()
    for root, dirs, files in os.walk(dir_name):
        for subdir in dirs:
            for sub_root, subdirs, subfiles in os.walk(os.path.join(root, subdir)):
                if sub_root in set_names:
                    print("%s 重复" % sub_root)
                set_names.add(sub_root)
                if not os.path.exists(sub_root.replace("_160", "")):
                    print(sub_root + " 不存在")
    print(len(set_names))


if __name__ == '__main__':
    find_diff()
import os
import shutil

org_dir_name = 'D:\\education\\CASIA_V5\\CASIA-FACEV5_160'
female_dir_name = 'D:\\education\\CASIA_V5\\CASIA-FACEV5_160_FEMALE'
target_male_dir = 'D:\\education\\CASIA_V5\\CASIA-FACEV5_160_MALE'


def copy_male():
    for root, dirs, files in os.walk(org_dir_name):
        for subdir in dirs:
            if os.path.isdir(os.path.join(female_dir_name, subdir)):
                print(subdir + '是女性')
                continue
            else:
                # copy male folder
                if os.path.isdir(os.path.join(target_male_dir, subdir)):
                    print("已拷贝")
                    continue
                shutil.copytree(os.path.join(root, subdir), os.path.join(target_male_dir, subdir))
                print('拷贝男性目录成功:' + subdir)


def check_dup():
    for root, dirs, files in os.walk(female_dir_name):
        for subdir in dirs:
            if os.path.isdir(os.path.join(target_male_dir, subdir)):
                print('重复' + subdir)
                shutil.rmtree(os.path.join(target_male_dir, subdir))


if __name__ == '__main__':
    #copy_male()
    check_dup()

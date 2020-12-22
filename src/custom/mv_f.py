import os
import shutil

pairs_filename = 'E:\\scrawl_images\\star_images_160\\f.txt'
dst_root = 'E:\\scrawl_images\\f_160\\'
pairs = []

def mv_to_f():
    with open(pairs_filename, 'r', encoding='UTF-8') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            if len(pair) != 3:
                print('长度不对')
                continue
            cand_file = pair[0]
            dst_folder = dst_root + cand_file[cand_file.rfind('\\'):cand_file.rfind('_')]
            if not os.path.exists(dst_folder):
                os.mkdir(dst_folder)
                pass
            print(dst_folder + cand_file[cand_file.rfind('\\'):])
            if os.path.exists(dst_folder + cand_file[cand_file.rfind('\\'):]):
                print(dst_folder + cand_file[cand_file.rfind('\\'):cand_file.rfind('_')] + '已存在')
                continue
            print(cand_file + '  ' + dst_folder)
            #pairs.append(pair)
            shutil.move(cand_file, dst_folder)


def cp_to_org():
    '''
        把原始图片移动回去
    '''
    src_root = 'E:\\scrawl_images\\f_160\\'
    dst_root = 'E:\\scrawl_images\\star_images_160\\'
    for root, dirs, files in os.walk(src_root):
        for subdir in dirs:
            for sub_root, subdirs, subfiles in os.walk(os.path.join(root, subdir)):
                if not os.path.exists(dst_root + subdir):
                    print('创建: ' + dst_root + subdir)
                    os.mkdir(dst_root + subdir)
                for sub_file in subfiles:
                    src_file = os.path.join(sub_root, sub_file)
                    dst_file = dst_root + subdir + os.sep + sub_file
                    print('原图片:' + src_file + ', 目的: ' + dst_file)
                    # 把图片复制回去
                    shutil.copy(src_file, dst_file)


if __name__ == '__main__':
    cp_to_org()
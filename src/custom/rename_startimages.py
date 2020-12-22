import os


def rename():
    dir_name = 'D:\\scrawl_images\\images2_raw'
    need_remove_dirs = []
    i = 0
    for root, dirs, files in os.walk(dir_name):
        for subdir in dirs:
            for sub_root, subdirs, subfiles in os.walk(os.path.join(root, subdir)):
                for img_filename in subfiles:
                    src = os.path.join(sub_root, img_filename)
                    # 把图片名称都转换成%d格式的
                    formatted_num = img_filename[img_filename.find('_')+1:img_filename.find('.')]
                    num = int(formatted_num)
                    dst_num = '%04d' % num
                    dst_name = subdir + '_' + img_filename.replace(formatted_num, dst_num)
                    dst_name = os.path.join(sub_root, dst_name)
                    os.rename(src, dst_name)
                    print("src:%s, dst:%s" % (src, dst_name))


if __name__ == '__main__':
    rename()
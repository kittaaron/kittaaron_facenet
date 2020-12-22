import compare
import os


if __name__ == '__main__':
    need_remove_dirs = []
    dir_name = 'E:\scrawl_images\star_images_160'
    i = 0
    for root, dirs, files in os.walk(dir_name):
        for subdir in dirs:
            argv = ["E:\\education\ML\\models\\20180402-114759"]
            for sub_root, subdirs, subfiles in os.walk(os.path.join(root, subdir)):
                new_subfiles = [os.path.join(sub_root, subfile) for subfile in subfiles]
                argv = argv + (new_subfiles)
            i += 1
            print("第%d个:%s" % (i, sub_root))
            parsed_args = compare.parse_arguments(argv)
            compare.main(parsed_args, sub_root)
# 筛选掉不合适的照片
import align.detect_face
from scipy import misc
import facenet
import os
import tensorflow as tf
import shutil

import os
import face_recognition
from PIL import Image
from PIL import ImageFile
from PIL import UnidentifiedImageError
import threading
import time
import cv2
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


def process_org_img(path, new_path):
    dirs = os.listdir(path)
    for pic_dir in dirs:
        print(pic_dir)
        dir_path = os.path.join(path, pic_dir)
        pics = os.listdir(dir_path)
        for pic in pics:
            new_pic_path = os.path.join(new_path, pic_dir)
            org_path = os.path.join(dir_path, pic)
            if os.path.exists(new_pic_path + '\\' + pic):
                # 把原图片删除
                os.remove(org_path)
                print('已转换过' + new_pic_path + '\\' + pic + ',删除原图:' + org_path)
                continue
            image = face_recognition.load_image_file(org_path)
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) == 0:
                print('%s 未检测到人脸' % pic)
                continue
            if len(face_locations) > 1:
                print('%s 检测到多张人脸' % pic)
                continue
            img = Image.open(org_path)
            # 如果图片小于160*160，移动到raw
            imgSize = img.size  # 大小/尺寸
            w = img.width  # 图片的宽
            h = img.height  # 图片的高
            f = img.format  # 图像格式
            if w < 160 or h < 160:
                print("宽高小于160:" + org_path)
                continue

            if not os.path.exists(new_pic_path):
                os.makedirs(new_pic_path)
            if len(img.split()) == 4:
                # 利用split和merge将通道从四个转换为三个
                r, g, b, a = img.split()
                toimg = Image.merge("RGB", (r, g, b))
                toimg.save(new_pic_path + '\\' + pic)
                img.close()
                os.remove(org_path)
                print('%s 转换通道保存成功,删除原图:%s' % (pic, org_path))
            else:
                try:
                    img.save(new_pic_path + '\\' + pic)
                    os.remove(org_path)
                    img.close()
                    print('%s save成功,删除原图:%s' % (pic, org_path))
                except:
                    continue
        print('Finish......!')


def lock_test(path, new_path):
    mu = threading.Lock()
    if mu.acquire(True):
        process_org_img(path, new_path)
        mu.release()


def check_face_subdir(subdir_abs_path, simple_pic_dir, dst_path):
    dst_dir_path = os.path.join(dst_path, simple_pic_dir)
    pics = os.listdir(subdir_abs_path)
    for pic in pics:
        org_path = os.path.join(subdir_abs_path, pic)
        dst_org_path = os.path.join(dst_dir_path, pic)

        image = face_recognition.load_image_file(org_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            print("未检测到人脸:" + org_path, ",移动到新的:" + dst_org_path)
            shutil.move(org_path, dst_org_path)
        if len(face_locations) > 1:
            print("检测到多张人脸:" + org_path, ",移动到新的:" + dst_org_path)
            shutil.move(org_path, dst_org_path)

    print('Finish......!')


def mv_abnormal_to_org():
    path = "D:\\scrawl_images\\images2"
    dst_path = "D:\\scrawl_images\\images2_raw"
    dirs = os.listdir(path)
    for simple_pic_dir in dirs:
        print(simple_pic_dir)
        subdir_abs_path = os.path.join(path, simple_pic_dir)
        check_face_subdir(subdir_abs_path, simple_pic_dir, dst_path)


def mv_abnormal_from_log_to_org():
    maybe_abnormal_file = 'D:\\scrawl_images\\may_be_abnormal.txt'
    moved = 0
    with open(maybe_abnormal_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            if len(pair) != 3:
                print('长度不对')
                continue
            abnormal_file = pair[0]
            if not os.path.exists(abnormal_file):
                print(abnormal_file + ' 图片不存在')
                continue
            dst_org_path = abnormal_file.replace('images2', 'images2_raw')
            print('异常图片:' + abnormal_file + ',移动到:' + dst_org_path)
            shutil.move(abnormal_file, dst_org_path)
            moved += 1
    print('总计移动异常图片:' + str(moved))


def check_facenum(path):
    image = face_recognition.load_image_file(path)
    face_locations = face_recognition.face_locations(image)
    print(len(face_locations))


def count_total(path):
    dirs = os.listdir(path)
    total = 0
    for pic_dir in dirs:
        dir_path = os.path.join(path, pic_dir)
        if not os.path.isdir(dir_path):
            print(dir_path + ' 不是一个目录')
            continue
        pics = os.listdir(dir_path)
        total += len(pics)
    print("total: %d" % total)


def check_if_true(org_pic_tuple, candid_tuple_arr):
    if len(candid_tuple_arr) <= 0:
        return
    org_pic_path = org_pic_tuple[0]
    org_encodings = org_pic_tuple[1]
    avg_threshold = 0
    true_num = 0
    false_num = 0
    # org_pic = face_recognition.load_image_file(org_pic_path)
    # org_encodings_arr = face_recognition.face_encodings(org_pic)
    # if len(org_encodings_arr) == 0:
    #     print(org_pic_path + '原图未检测到图片')
    #     return
    #org_encodings = org_encodings_arr[0]
    for candid_tuple in candid_tuple_arr:
        candid_path = candid_tuple[0]
        candid_encodings = candid_tuple[1]
        #print('candid:' + candid_path)
        # condid_pic = face_recognition.load_image_file(candid_path)
        # candid_encodings_arr = face_recognition.face_encodings(condid_pic)
        # if len(candid_encodings_arr) == 0:
        #     print(candid_path + '目标图未检测到图片')
        #     continue
        # candid_encodings = candid_encodings_arr[0]
        #results = face_recognition.compare_faces([org_encodings], candid_encodings, tolerance=0.55)
        face_distances = face_recognition.face_distance([org_encodings], candid_encodings)
        print(org_pic_path + '和' + candid_path + '距离为:' + str(face_distances))
        ##### 相同图片的处理
        if face_distances <= 0.20:
            print(org_pic_path + ' 和 ' + candid_path + ' 可能是同一张图片')
            s_file = open(os.path.join("D:\\scrawl_images", 'may_be_same.txt'), 'a', encoding='utf-8')
            s_file.write(org_pic_path + "\t" + candid_path + "\n")
            s_file.close()
        #if results[0]:
        if face_distances <= 0.51:
            true_num += 1
        else:
            false_num += 1
    if (false_num / (false_num + true_num)) >= 0.5:
    #if false_num >= true_num:
        # 写入到文件中
        f_file = open(os.path.join("D:\\scrawl_images", 'may_be_abnormal.txt'), 'a', encoding='utf-8')
        f_file.write(org_pic_path + "\t" + str(false_num) + "\t" + str(true_num) + "\n")
        f_file.close()
        print(org_pic_path + '可能是噪声图片.')
    else:
        print(org_pic_path + '应该是正确图片.')


def hanles_pics(pics):
    for i, pic_tuple in enumerate(pics):
        candid_tuple_arr = pics[:i]
        candid_tuple_arr.extend(pics[i+1:])
        check_if_true(pic_tuple, candid_tuple_arr)


def traverse_root_find_ab(root_dir):
    '''
    遍历图片，找出同一目录中可疑图片
    '''
    #root_dir = "D:\\scrawl_images\\images2_raw"
    dirs = os.listdir(root_dir)
    total = 0
    for pic_dir in dirs:
        sub_dir_path = os.path.join(root_dir, pic_dir)
        traverse_subdir_find_ab(sub_dir_path, pic_dir)


def traverse_subdir_find_ab(subdir_abs_path, simple_pic_dir):
    print(subdir_abs_path + ' 开始处理')
    if not os.path.isdir(subdir_abs_path):
        print(subdir_abs_path + ' 不是文件夹')
        return
    start_second = time.time()
    simple_pics = os.listdir(subdir_abs_path)
    absolute_pics = []
    dst_dir_path = os.path.join('D:\\scrawl_images\\dup\\images2_160_ab', simple_pic_dir)
    if os.path.exists(dst_dir_path):
        print('已处理过')
        return
    f_file = open(os.path.join("D:\\scrawl_images\\dup", '160_abnormal.txt'), 'a', encoding='utf-8')
    for simple_pic in simple_pics:
        #subdir_abs_path = os.path.join(subdir_abs_path, simple_pic)

        org_path = os.path.join(subdir_abs_path, simple_pic)
        dst_org_path = os.path.join(dst_dir_path, simple_pic)

        # image = face_recognition.load_image_file(org_path)
        # face_locations = face_recognition.face_locations(image)
        # if len(face_locations) == 0:
        #     print("未检测到人脸:" + org_path, ",移动到新的:" + dst_org_path)
        #     shutil.move(org_path, dst_org_path)
        #     continue
        # if len(face_locations) > 1:
        #     print("检测到多张人脸:" + org_path, ",移动到新的:" + dst_org_path)
        #     shutil.move(org_path, dst_org_path)
        #     continue

        condid_pic = face_recognition.load_image_file(org_path)
        candid_encodings_arr = face_recognition.face_encodings(condid_pic)
        if len(candid_encodings_arr) == 0:
            print(org_path + '目标图未检测到人脸,移动到raw目录')
            if not os.path.exists(dst_dir_path):
                os.mkdir(dst_dir_path)
            shutil.move(org_path, dst_org_path)
            #shutil.copy(org_path, dst_org_path)
            f_file.write(org_path + "\n")
            continue
        if len(candid_encodings_arr) > 1:
            print(org_path + '目标图检测到多张人脸,移动到raw目录')
            if not os.path.exists(dst_dir_path):
                os.mkdir(dst_dir_path)
            shutil.move(org_path, dst_org_path)
            #shutil.copy(org_path, dst_org_path)
            f_file.write(org_path + "\n")
            continue
        absolute_pics.append((org_path, candid_encodings_arr[0]))

    f_file.close()
    #hanles_pics(absolute_pics)
    end_second = time.time()
    print(subdir_abs_path + '耗时' + str(end_second - start_second))


def find_too_small():
    path = "D:\\scrawl_images\\images2"
    dst_path = "D:\\scrawl_images\\images2_raw"
    dirs = os.listdir(path)
    for pic_dir in dirs:
        print(pic_dir)
        dir_path = os.path.join(path, pic_dir)
        dst_dir_path = os.path.join(dst_path, pic_dir)
        pics = os.listdir(dir_path)
        for pic in pics:
            org_path = os.path.join(dir_path, pic)
            dst_org_path = os.path.join(dst_dir_path, pic)
            try:
                img = Image.open(org_path)
                # 如果图片小于160*160，移动到raw
                imgSize = img.size  # 大小/尺寸
                w = img.width  # 图片的宽
                h = img.height  # 图片的高
                f = img.format  # 图像格式
                img.close()
                if w < 160 or h < 160:
                    print("宽高小于160:" + org_path, ",移动到新的:" + dst_org_path)
                    shutil.move(org_path, dst_org_path)
            except UnidentifiedImageError as e:
                img.close()
                print("识别不了图片:" + org_path, ",移动到新的:" + dst_org_path)
                shutil.move(org_path, dst_org_path)
        print('Finish......!')


def del_abnormals():
    abnormal_filename = 'D:\\scrawl_images\\abnormal_1.3.txt'
    with open(abnormal_filename, 'r', encoding='GBK') as f:
        for line in f.read().splitlines():
            if os.path.exists(line):
                shutil.rmtree(line)
                print('删除目录OK：%s' % line)
            if os.path.exists(line.replace('_160', '')):
                shutil.rmtree(line.replace('_160', ''))
                print('删除目录OK：%s' % line.replace('_160', ''))
            else:
                print("%s 不存在" % line)



def detec_face_num():
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    need_remove_dirs = []
    i = 0
    dir_name = 'D:\\scrawl_images\\star_images\\'
    for root, dirs, files in os.walk(dir_name):
        for subdir in dirs:
            for sub_root, subdirs, subfiles in os.walk(os.path.join(root, subdir)):
                i += 1
                if i % 10 == 0:
                    print('第%d个目录' % i)
                imgi_num = len(subfiles)
                multiface_imgpath = []
                print('"%s" %d张图' % (subdir, imgi_num))
                for img_filename in subfiles:
                    img_path = os.path.join(sub_root, img_filename)
                    try:
                        img = misc.imread(img_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(img_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % img_path)
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:, :, 0:3]

                        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                                          factor)
                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces < 1:
                            print('no face detected "%s"' % img_path)
                            os.remove(img_path)
                            continue
                        elif nrof_faces > 1:
                            print('multi face detected "%s"' % img_path)
                            multiface_imgpath.append(img_path)
                            os.remove(img_path)
                            continue

                multiface_img_num = len(multiface_imgpath)
                if imgi_num < 3:
                    print("%s 目录只有 %d 张图片" % (sub_root, imgi_num))
                    shutil.rmtree(sub_root)
                    continue
                if (multiface_img_num / imgi_num) > 0.5:
                    print('%s总共%d张图片,%d张有多张人脸' % (sub_root, imgi_num, multiface_img_num))
                    need_remove_dirs.append(sub_root)

    print('需要删除的目录:%s' % need_remove_dirs)


def del_dirs():
    dirs = ['D:\\scrawl_images\\images\\丁彦雨航', 'D:\\scrawl_images\\images\\丁霞', 'D:\\scrawl_images\\images\\于虹',
            'D:\\scrawl_images\\images\\傅娟', 'D:\\scrawl_images\\images\\凤凰传奇', 'D:\\scrawl_images\\images\\刘洵',
            'D:\\scrawl_images\\images\\南征北战NZBZ', 'D:\\scrawl_images\\images\\卢鑫', 'D:\\scrawl_images\\images\\卯秋民',
            'D:\\scrawl_images\\images\\史鸿飞', 'D:\\scrawl_images\\images\\叶良辰', 'D:\\scrawl_images\\images\\吉喆',
            'D:\\scrawl_images\\images\\唐才育', 'D:\\scrawl_images\\images\\孙丽雅', 'D:\\scrawl_images\\images\\孙明明',
            'D:\\scrawl_images\\images\\孙继海', 'D:\\scrawl_images\\images\\孟小冬', 'D:\\scrawl_images\\images\\宋喆',
            'D:\\scrawl_images\\images\\尼成', 'D:\\scrawl_images\\images\\崔文彩', 'D:\\scrawl_images\\images\\张伟丽',
            'D:\\scrawl_images\\images\\张玉浩', 'D:\\scrawl_images\\images\\徐明朝', 'D:\\scrawl_images\\images\\旭日阳刚',
            'D:\\scrawl_images\\images\\易建联', 'D:\\scrawl_images\\images\\曾繁日', 'D:\\scrawl_images\\images\\李乘德',
            'D:\\scrawl_images\\images\\李好', 'D:\\scrawl_images\\images\\李少芬', 'D:\\scrawl_images\\images\\李春江',
            'D:\\scrawl_images\\images\\李景亮', 'D:\\scrawl_images\\images\\杜锋', 'D:\\scrawl_images\\images\\杨九郎',
            'D:\\scrawl_images\\images\\杨威', 'D:\\scrawl_images\\images\\杨雨辰', 'D:\\scrawl_images\\images\\杭盖乐队',
            'D:\\scrawl_images\\images\\王九龙', 'D:\\scrawl_images\\images\\王倩一', 'D:\\scrawl_images\\images\\王哲林',
            'D:\\scrawl_images\\images\\王嘉诚', 'D:\\scrawl_images\\images\\王声', 'D:\\scrawl_images\\images\\王少杰',
            'D:\\scrawl_images\\images\\王治郅', 'D:\\scrawl_images\\images\\王适娴', 'D:\\scrawl_images\\images\\玖月奇迹',
            'D:\\scrawl_images\\images\\田明鑫', 'D:\\scrawl_images\\images\\白慧明', 'D:\\scrawl_images\\images\\祝铭震',
            'D:\\scrawl_images\\images\\穆铁柱', 'D:\\scrawl_images\\images\\羽泉', 'D:\\scrawl_images\\images\\艾慧',
            'D:\\scrawl_images\\images\\蒲熠星', 'D:\\scrawl_images\\images\\袁心玥', 'D:\\scrawl_images\\images\\许婧',
            'D:\\scrawl_images\\images\\许海峰', 'D:\\scrawl_images\\images\\谢金', 'D:\\scrawl_images\\images\\谭凯',
            'D:\\scrawl_images\\images\\贾云馨', 'D:\\scrawl_images\\images\\贾岳川', 'D:\\scrawl_images\\images\\赖文峰',
            'D:\\scrawl_images\\images\\赵欣培', 'D:\\scrawl_images\\images\\赵睿', 'D:\\scrawl_images\\images\\赵继伟',
            'D:\\scrawl_images\\images\\路金波', 'D:\\scrawl_images\\images\\郑海霞', 'D:\\scrawl_images\\images\\阴三儿',
            'D:\\scrawl_images\\images\\阿里郎组合', 'D:\\scrawl_images\\images\\陈善宝', 'D:\\scrawl_images\\images\\陈坚红',
            'D:\\scrawl_images\\images\\陈招娣', 'D:\\scrawl_images\\images\\雷庆瑶', 'D:\\scrawl_images\\images\\韩德君',
            'D:\\scrawl_images\\images\\高诗岩', 'D:\\scrawl_images\\images\\魏秋月', 'D:\\scrawl_images\\images\\鲍天琦']
    for dirname in dirs:
        print('删除文件:%s' % dirname)
        shutil.rmtree(dirname)


def del_toosmall():
    dir_name = 'D:\\scrawl_images\\images2_160\\'
    need_remove_dirs = []
    i = 0
    total = 0
    for root, dirs, files in os.walk(dir_name):
        for subdir in dirs:
            for sub_root, subdirs, subfiles in os.walk(os.path.join(root, subdir)):
                total += len(subfiles)
                if len(subfiles) <= 0:
                    print('%s 只有 %d 张图片' % (subdir, len(subfiles)))
                    src_dir = os.path.join(root, subdir)
                    dst_dir = src_dir.replace('images2', 'images2_raw')
                    print(src_dir + ' -> ' + dst_dir)
                    i+=1
                    #shutil.move(src_dir, dst_dir)
                    shutil.rmtree(src_dir)
    print("筛选完成.图片总数: %s, 移动数量: %d" % (total, i))


def mv_abnormal_13():
    abnormal_filename = 'D:\\scrawl_images\\abnormal_1.3.txt'
    dst_dir_160 = "D:\\scrawl_images\\abnormal_1.3\\160"
    dst_dir = "D:\\scrawl_images\\abnormal_1.3\\org"
    with open(abnormal_filename, 'r', encoding='GBK') as f:
        for line in f.read().splitlines():
            if os.path.exists(line):
                shutil.move(line, dst_dir_160)
                print('移动160OK：%s' % line)
            if os.path.exists(line.replace('_160', '')):
                shutil.move(line.replace('_160', ''), dst_dir)
                print('移动OK：%s' % line.replace('_160', ''))
            else:
                print("%s 不存在" % line)


def mv_abnormal_12():
    abnormal_filename = 'D:\\scrawl_images\\abnormal_1.2.txt'
    dst_dir_160 = "D:\\scrawl_images\\abnormal_1.2\\160"
    dst_dir = "D:\\scrawl_images\\abnormal_1.2\\org"
    with open(abnormal_filename, 'r', encoding='GBK') as f:
        for line in f.read().splitlines():
            if os.path.exists(line):
                shutil.move(line, dst_dir_160)
                print('移动160OK：%s' % line)
            if os.path.exists(line.replace('_160', '')):
                shutil.move(line.replace('_160', ''), dst_dir)
                print('移动OK：%s' % line.replace('_160', ''))
            else:
                print("%s 不存在" % line)


def mv_same_from_log_to_org():
    maybe_abnormal_file = 'D:\\scrawl_images\\may_be_same.txt'
    moved = 0
    total_dict = {}
    with open(maybe_abnormal_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split('\t')
            if len(pair) != 2:
                print('长度不对')
                continue
            same_pair_1 = pair[0]
            same_pair_2 = pair[1]
            dirname = same_pair_1[same_pair_1.rfind('\\'):same_pair_1.rfind('_')]
            if dirname not in total_dict:
                total_dict[dirname] = []
            small = min(same_pair_1, same_pair_2)
            big = max(same_pair_1, same_pair_2)
            if (small, big) not in total_dict[dirname]:
                total_dict[dirname].append((small, big))

    #print(total_dict)
    total_moved = 0
    have_same_pics_persons = 0
    for (k, tuple_list) in total_dict.items():
        same_size = len(tuple_list)
        print("处理%s,共%d组" % (k, len(tuple_list)))
        i = 0
        for small, big in tuple_list:
            #small = tuple_i[0]
            #big = tuple_i[1]
            if os.path.exists(big):
                # 如果还没被移走
                if not os.path.exists(big):
                    print(big + ' 图片不存在，可能已被移走')
                    continue
                dst_big_path = big.replace('images2', 'images2_raw')
                print('相同图片big:' + big + ',移动到:' + dst_big_path)
                shutil.move(big, dst_big_path)
                total_moved += 1
            else:
                # 可能之前的某一对出现，已经被移走了，这种情况，把small也移走
                if i == 0:
                    # 如果只有一对相同的，防止唯一的一张也被移走
                    pass
                else:
                    if not os.path.exists(small):
                        print(small + ' 图片2不存在，可能已被移走')
                        continue
                    dst_small_path = small.replace('images2', 'images2_raw')
                    print('相同图片small:' + small + ',移动到:' + dst_small_path)
                    shutil.move(small, dst_small_path)
                    total_moved += 1
            i+=1
        have_same_pics_persons += 1
    print("总人数：%d 共移除相同图片：%d" % (have_same_pics_persons, total_moved))


def mv_dir():
    arr = ['VAVA','丁俊晖','乔乔','于洋','于虹','井元林','代煜龙','何伟','傅娟','光光','关东','关少曾','关山','冯宝宝','冷弦','凤凰传奇','凯迪','刘丹','刘云','刘云天','刘天骐','刘宝瑞','刘晓莉','刘波','刘畅','刘轩豪','南征北战NZBZ','卢嘉丽','卢奇','卢金聪','卯秋民','古筝','叶敬林','吴晴晴','吴永宁','吴永强','周文','周瑜','唐诗逸','壮丽','姜桂成','孔德宝','孙云玲','孙元良','孙迅','孟小冬','宋喆','宝木中阳','宝石Gem','小乔','小凌','小爱','小贱','少司命','尤奕','尧十三','尼成','崔健','崔文彩','左小祖咒','常宝堃','平安','张彻','张浩','张立','张良','张艺洋','徐明朝','徐枫','成天龙','房映华','方逸华','旭日阳刚','时诗','曹轩宾','朱传宇','李丹宁','李乘德','李元霸','李宁','李文华','李明','李楠','李正春','李毅','李燃','李瑞','李紫涵','李肖逸','李菁','李越昕蕾','李飞','杜鹃','杨宁','杨明','杨梅','杭盖乐队','格格','梁佳玉','梁欢','梁焯满','梁龙','江南','汪鑫洁','法老','洛桑·尼玛','洪尧','洪涛','海伦','潘梦莹','牛群','狄娜','王东','王亚楠','王伟','王嘉诚','王天林','王婧','王小虎','王岩','王建国','王慧','王时雨','王洋','王良','王超','王辉','王阳明','玖月奇迹','石坚','石天','石头','祝铭震','福克斯','秦祥林','窦唯','窦颖','红薇','维妮','罗琦','罗秀春','胡小玲','舒建臣','艾菲','许婧','许文赫','谈莉娜','谢金','谭永华','贾岳川','赵丹','赵丽娟','赵照','那吾克热·玉素甫江','郑云','郑成华','郑文喜','郭峰','郭汾瑒','闰土','阴三儿','阿尔法','阿里郎组合','陈友','陈善宝','陈坚红','陈奕雯','陈强','陈招娣','陈旭','陈浩','陈盛桐','陈超','韦唯','韩克','马丽','马超','马頔','高大伟','黄忠东','黄新','黄旭','黎明','黑龙']
    for name_i in arr:
        src_dir = 'D:\\scrawl_images\\images2_raw\\' + name_i + '\\' + name_i
        # dst_dir = 'D:\\scrawl_images\\images2_raw\\' + name_i
        # pics = os.listdir(src_dir)
        # for pic in pics:
        #     src_file = os.path.join(src_dir, pic)
        #     dst_file = os.path.join(dst_dir, pic)
        #     print(src_file + ' -> ' + dst_file)
        #     shutil.move(src_file, dst_file)
        shutil.rmtree(src_dir)
        #os.remove(src_dir)


def del_images2_ab():
    '''
        删除转换之后的异常图片
    '''
    ab_path = "D:\\education\\CASIA_V5\\CASIA-FACEV5_160"
    #del_path = "D:\\scrawl_images\\images2_160"
    dirs = os.listdir(ab_path)
    total_ab = 0
    for pic_dir in dirs:
        print(pic_dir)
        ab_dir_path = os.path.join(ab_path, pic_dir)
        #del_sub_path = os.path.join(del_path, pic_dir)
        pics = os.listdir(ab_dir_path)
        for pic in pics:
            ab_file_path = os.path.join(ab_dir_path, pic)
            #del_file_path = os.path.join(del_sub_path, pic)
            #img = cv2.imread(ab_file_path)
            cv_img = cv2.imdecode(np.fromfile(ab_file_path, dtype=np.uint8), -1)
            face = cv2.CascadeClassifier('D:\\git\\facenet\\haarshare\\haarcascade_frontalface_default.xml')  # 创建人脸检测器    放在同目录
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)  # 将img转为回复图像，存放中gray中
            faces = face.detectMultiScale(gray, 1.1, 3)  # 检测图像中的人脸
            if len(faces) >= 1:
                print('发现人脸:' + ab_file_path)
            else:
                print('未发现人脸:' + ab_file_path)
                total_ab += 1

            #print('准备删除图片:' + del_file_path)
            #os.remove(del_file_path)
    print("总计异常 %d" % total_ab)


def restore_mannully_ab():
    '''
        恢复手动移除的异常图片 -> 以比较训练结果
    '''
    ab_path = "D:\\scrawl_images\\images2_160_ab"
    del_path = "D:\\scrawl_images\\images2_160"
    maybe_abnormal_file = 'D:\\scrawl_images\\may_be_abnormal.txt'

    auto_ab = []
    with open(maybe_abnormal_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            if len(pair) != 3:
                print('长度不对')
                continue
            ab_file = pair[0].replace('images2', 'images2_160')
            auto_ab.append(ab_file)
    print("自动生成的数量: %d" % len(auto_ab))

    dirs = os.listdir(ab_path)
    mannuled = 0
    autoed = 0
    s_file = open(os.path.join("D:\\scrawl_images", 'mannul_remove.txt'), 'w', encoding='utf-8')

    for pic_dir in dirs:
        print(pic_dir)
        ab_dir_path = os.path.join(ab_path, pic_dir)
        restore_sub_path = os.path.join(del_path, pic_dir)
        pics = os.listdir(ab_dir_path)
        for pic in pics:
            ab_file_path = os.path.join(ab_dir_path, pic)
            restore_file_path = os.path.join(restore_sub_path, pic)
            if restore_sub_path in auto_ab:
                print(restore_sub_path + ' 属于自动识别的')
                autoed += 1
            else:
                mannuled += 1
            #img = cv2.imread(ab_file_path)
            # cv_img = cv2.imdecode(np.fromfile(del_file_path, dtype=np.uint8), -1)
            # face = cv2.CascadeClassifier('D:\\git\\facenet\\haarshare\\haarcascade_frontalface_default.xml')  # 创建人脸检测器    放在同目录
            # gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)  # 将img转为回复图像，存放中gray中
            # faces = face.detectMultiScale(gray, 1.1, 3)  # 检测图像中的人脸
            # if len(faces) >= 1:
            #     print('发现人脸:' + del_file_path)
            # else:
            #     print('未发现人脸:' + del_file_path)

            print('准备恢复图片:' + restore_file_path)
            # 写入文件，已被重新删除
            s_file.write(restore_file_path + "\n")
    s_file.close()
    print('自动删除图片: %d, 手动删除: %d' % (autoed, mannuled))


def start_file():
    path_160 = "D:\\scrawl_images\\images2_160"
    ab_path_160 = "D:\\scrawl_images\\images2_160_ab"
    dirs = os.listdir(path_160)
    i = 0
    for pic_dir in dirs:
        i+=1
        print(i)
        if i <= 2548:
            continue
        dir_path = os.path.join(path_160, pic_dir)
        ab_file_path = os.path.join(ab_path_160, pic_dir)
        if not os.path.isdir(dir_path):
            continue
        if not os.path.isdir(ab_file_path):
            os.mkdir(ab_file_path)
        os.startfile(ab_file_path)
        os.startfile(dir_path)
        input()


def find_wrong_pair():
    maybe_abnormal_file = 'D:\\scrawl_images\\images2_160\\pairs.txt'
    records = []
    with open(maybe_abnormal_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines()[1:]:
            records.append(line)
    print("re: %d" % len(records))

    result_arr = [

         [True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, False, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False],
         [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
         [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, False, True, False, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False],
         [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True],
         [True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
         [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
         [True, True, True, True, True, True, False, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, True, False, False, False, False, False, False],
         [True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
         [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False],
         [False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False]

    ]

    r_f_index = []
    f_f_index = []
    cycle = 0
    for result_i in result_arr:
        i = 0
        i_r_f = []
        i_f_f = []
        for r in result_i:
            if (int(i / 300)) % 2 == 0:
                if not r:
                    i_r_f.append(cycle * 600 + i)
                    r_f_index.append(cycle * 600 + i)
            else:
                if r:
                    i_f_f.append(cycle * 600 + i)
                    f_f_index.append(cycle * 600 + i)
            i+=1
        cycle += 1
        print('i_r_f: %s, i_f_f: %s' % (i_r_f, i_f_f))
        print('i: %d, 正例错误: %d, 反例识别错误: %d' % (i, len(i_r_f), len(i_f_f)))
    print('r_f_index: %s, f_f_index: %s' % (r_f_index, f_f_index))
    root_dir = 'D:\\scrawl_images\\images2_160'
    root_ab_dir = 'D:\\scrawl_images\\images2_160_ab'
    s_file = open(os.path.join("D:\\scrawl_images", 'wrong_pair.txt'), 'w', encoding='utf-8')
    for temp in r_f_index:
        s_file.write(records[temp])
    for temp2 in f_f_index:
        s_file.write(records[temp2])
    s_file.close()
    for id_i_r in r_f_index:
        print(records[id_i_r])
        pair = records[id_i_r].strip().split()
        people_name = pair[0]
        p_1 = people_name + '_' + pair[1]
        p_2 = people_name + '_' + pair[2]
        people_dir = os.path.join(root_dir, people_name)
        ab_dir = os.path.join(root_ab_dir, people_name)
        path_1 = os.path.join(people_dir, p_1) + '.png'
        path_2 = os.path.join(people_dir, p_2) + '.png'
        os.startfile(people_dir)
        os.startfile(ab_dir)
        input()
    for id_i_f in f_f_index:
        print(records[id_i_f])
        pair = records[id_i_f].strip().split()
        people_1_name = pair[0]
        people_2_name = pair[2]
        p_1 = people_1_name + '_' + pair[1]
        p_2 = people_2_name + '_' + pair[3]
        people_1_dir = os.path.join(root_dir, people_1_name)
        people_2_dir = os.path.join(root_dir, people_2_name)
        path_1 = os.path.join(people_1_dir, p_1) + '.png'
        path_2 = os.path.join(people_2_dir, p_2) + '.png'
        os.startfile(people_1_dir)
        os.startfile(people_2_dir)
        input()



if __name__ == '__main__':
    # del_dirs()

    #del_toosmall()
    #start_file()
    #restore_mannully_ab()
    # del_abnormals()
    # mv_abnormal_12()
    """
    # 这一段是用来筛选mages2_raw中的不合格照片
    paths = [r'E:\scrawl_images\images2_raw']
    new_paths = [r'E:\scrawl_images\images2']
    for i in range(len(paths)):
        my_thread = threading.Thread(target=lock_test, args=(paths[i], new_paths[i]))
        my_thread.start()
    """
    #mv_dir()
    # find_too_small()
    #check_facenum("D:\\scrawl_images\\images2\\黄锦燊\\黄锦燊_0016.jpg")
    #count_total("D:\\scrawl_images\\images2_160")
    #del_images2_ab()
    #find_wrong_pair()
    #check_face_subdir('D:\\scrawl_images\\images2\\彭禺厶', '彭禺厶', 'D:\\scrawl_images\\images2_raw')
    #mv_abnormal_to_org()
    #traverse_root_find_ab('D:\\scrawl_images\\dup\\images2_160')
    #traverse_subdir_find_ab('D:\\scrawl_images\\dup\\images2_160\\Yamy', 'Yamy')
    #mv_abnormal_from_log_to_org()
    #mv_same_from_log_to_org()

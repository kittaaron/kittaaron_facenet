import glob
import os.path
import numpy as np
import os

# 图片数据文件夹
INPUT_DATA = 'D:\\scrawl_images\\images2_160\\'


def create_same_pairs():
    matched_result = set()
    k = 0
    # 获取当前目录下所有的子目录,这里x 是一个三元组(root,dirs,files)，第一个元素表示INPUT_DATA当前目录，
    # 第二个元素表示当前目录下的所有子目录,第三个元素表示当前目录下的所有的文件
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    match_num = 3000
    while len(matched_result) < match_num:
        for sub_dir in sub_dirs[1:]:
            if len(matched_result) >= match_num:
                break
            # 使用mtcnn预先生成的文件都是png后缀
            extensions = 'png'
            # 把单个人物图片存放在file_list列表里
            person_pics = []
            dir_name = os.path.basename(sub_dir)
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extensions)
            person_pics.extend(glob.glob(file_glob))
            if not person_pics: continue
            # 通过目录名获取类别的名称
            label_name = dir_name
            length = len(person_pics)
            random_number1 = np.random.randint(50)
            random_number2 = np.random.randint(50)
            base_name1 = os.path.basename(person_pics[random_number1 % length])  # 获取文件的名称
            base_name2 = os.path.basename(person_pics[random_number2 % length])

            if person_pics[random_number1 % length] != person_pics[random_number2 % length]:
                matched_result.add(label_name + '\t' + base_name1[base_name1.rfind('_')+1:base_name1.rfind('.')] + '\t' + base_name2[base_name2.rfind('_')+1:base_name2.rfind('.')])
                k += 1
                if k % 100 == 0:
                    print('len(match): %d' % len(matched_result))
                    print(k)
        if len(matched_result) >= match_num:
            break

    # 返回整理好的所有数据
    return matched_result, match_num


# 创建pairs.txt
def create_diff_pairs():
    unmatched_result = set()  # 不同类的匹配对
    k = 0
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # sub_dirs[0]表示当前文件夹本身的地址，不予考虑，只考虑他的子目录
    for sub_dir in sub_dirs[1:]:
        # 获取当前目录下所有的有效图片文件
        extensions = ['png']
        file_list = []
        # 把图片存放在file_list列表里

        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            # glob.glob(file_glob)获取指定目录下的所有图片，存放在file_list中
            file_list.extend(glob.glob(file_glob))

    length_of_dir = len(sub_dirs)
    print(length_of_dir)
    match_num = 3000
    for k in range(1000):
        for i in range(length_of_dir):
            if len(unmatched_result) >= match_num:
                break
            class1 = sub_dirs[i]
            random_num = np.random.randint(5000000)
            i2 = random_num % length_of_dir
            if i == i2:
                continue
            class2 = sub_dirs[i2]

            class1_name = os.path.basename(class1)
            class2_name = os.path.basename(class2)
            # 获取当前目录下所有的有效图片文件
            extensions = 'png'
            file_list1 = []
            file_list2 = []
            # 把图片存放在file_list列表里
            file_glob1 = os.path.join(INPUT_DATA, class1_name, '*.' + extension)
            file_list1.extend(glob.glob(file_glob1))
            file_glob2 = os.path.join(INPUT_DATA, class2_name, '*.' + extension)
            file_list2.extend(glob.glob(file_glob2))
            if file_list1 and file_list2:
                base_name1 = os.path.basename(file_list1[random_num % len(file_list1)])  # 获取文件的名称
                base_name2 = os.path.basename(file_list2[random_num % len(file_list2)])
                # unmatched_result.add([class1_name, base_name1, class2_name, base_name2])
                s = class2_name + '\t' + base_name2 + '\t' + class1_name + '\t' + base_name1
                if (s not in unmatched_result):
                    unmatched_result.add(class1_name + '\t' + base_name1[base_name1.rfind('_')+1:base_name1.rfind('.')] + '\t' + class2_name + '\t' + base_name2[base_name2.rfind('_')+1:base_name2.rfind('.')])
                k = k + 1
                if k % 100 == 0:
                    print(k)
        if len(unmatched_result) >= match_num:
            break

    return unmatched_result, match_num


result, k1 = create_same_pairs()
print(len(result))
# print(result)

result_un, k2 = create_diff_pairs()
print(len(result_un))
# print(result_un)

file = open(os.path.join(INPUT_DATA, 'pairs.txt'), 'w', encoding='utf-8')

result1 = list(result)
result2 = list(result_un)

file.write('10 300\n')

j = 0
for i in range(100):
    j = 0
    print("=============================================第" + str(i) + '次, 相同的')
    for pair in result1[i * 300:i * 300 + 300]:
        j = j + 1
        print(str(j) + ': ' + pair)
        file.write(pair + '\n')

    print("=============================================第" + str(i) + '次, 不同的')
    for pair in result2[i * 300:i * 300 + 300]:
        j = j + 1
        print(str(j) + ': ' + pair)
        file.write(pair + '\n')
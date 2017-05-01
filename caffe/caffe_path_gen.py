import os
import sys
import argparse
import numpy as np

current_path = os.path.abspath(os.path.dirname(__file__))

class_index_dict = {}
def generate_words(train_path):
    print(train_path)
    i = 0
    words_file = open('words.txt', 'w+')
    for fn in os.listdir(train_path):
        filepath = os.path.join(train_path, fn)
        if os.path.isdir(filepath):
            print('Find the {0} class: {1}'.format(i, fn))
            words_file.write(str(i) + ' ' + fn + '\n')
            class_index_dict[fn] = str(i)
            i = i+1

    words_file.close()
    print('All class index and names has been saved into words.txt!')
    print(class_index_dict)


def generate_train_images_path(train_path, is_shuffle):
    train_txt = open('train.txt', 'w+')

    all_train_lines = []
    for fn in os.listdir(train_path):
        filepath = os.path.join(train_path, fn)
        if os.path.isdir(filepath):
            for root, dirs, files in os.walk(filepath):
                for file in files:
                    imagefilepath = os.path.join(root, file)
                    print('{0} {1}'.format(imagefilepath, class_index_dict[fn]))
                    all_train_lines.append(imagefilepath + ' ' + class_index_dict[fn] + '\n')

    if is_shuffle:
        np.random.shuffle(all_train_lines)
        for line in all_train_lines:
            train_txt.write(line)
    else:
        for line in all_train_lines:
            train_txt.write(line)

    train_txt.close()
    print('All train images path and class index has been saved into test_images.txt!')



def generate_valid_images_path(test_path):
    valid_txt = open('valid.txt', 'w+')

    for fn in os.listdir(test_path):
        filepath = os.path.join(test_path, fn)
        if os.path.isdir(filepath):
            for root, dirs, files in os.walk(filepath):
                for file in files:
                    imagefilepath = os.path.join(root, file)
                    print('{0} {1}'.format(imagefilepath, class_index_dict[fn]))
                    valid_txt.write(imagefilepath + ' ' + class_index_dict[fn] + '\n')

    valid_txt.close()
    print('All test images path and class index has been saved into train_images.txt!')


def generate_test_images_path(valid_path):
    test_txt = open('test.txt', 'w+')
    for fn in os.listdir(valid_path):
        file_path = os.path.join(valid_path, fn)
        if os.path.isfile(file_path):
            print(file_path)
            test_txt.write(file_path + '\n')
    test_txt.close()



def parse_args():
    parse = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                description='This script using for Caffe generate words.txt and train images\
		paths, also test images paths. This will generate alongside this script directory.'
        )
    parse.add_argument('-train', type=str, help='your train images path, you can\
		place all your classes images into a train_images folder, this script can solve\
		it automatically.')
    parse.add_argument('-test', type=str, help='your test images path, place all your test\
		images into a test_images folder.')

    parse.add_argument('-valid', type=str, help='your valid images path this will not with label.')

    parse.add_argument('-shuffle', type=bool, default=False, help='Do shuffle your train images or not.')

    args = parse.parse_args()

    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    train_images_path = args['train']
    test_images_path = args['test']
    valid_images_path = args['valid']
    is_shuffle = args['shuffle']
    if os.path.isdir(args['train']):
        generate_words(train_images_path)
        generate_train_images_path(train_images_path, is_shuffle)
    else:
        print('Your train images path seems invalid, please check it and try again.')

    if test_images_path:
        if os.path.isdir(args['test']):
            generate_test_images_path(test_images_path)
        else:
            print('Your test images path seems invalid, please check it and try again.')

    if valid_images_path:
        if os.path.isdir(args['valid']):
            generate_valid_images_path(valid_images_path)
        else:
            print('Your valid images path seems invalid.')
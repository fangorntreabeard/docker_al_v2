import os
import json
import copy
import random


def write_json(list_files, m):
    full_train_json = '/home/neptun/PycharmProjects/datasets/coco/instances_train2017.json'
    current_label = 17  # cat

    with open(full_train_json) as f:
        razmetka = json.load(f)

    categories = razmetka['categories']
    annotations = razmetka['annotations']
    images = razmetka['images']
    info = razmetka['info']
    licenses = razmetka['licenses']

    new_image = []
    a = []
    for row in images:
        if row['file_name'] in list_files:
            a.append(row['id'])
            copy_row = copy.deepcopy(row)
            new_image.append(copy_row)

    new_annotation = []
    for row in annotations:
        if row['category_id'] == current_label and row['image_id'] in a:
            copy_row = copy.deepcopy(row)
            copy_row['segmentation'] = []
            new_annotation.append(copy_row)

    b = []
    for row in new_annotation:
        b.append(row['image_id'])

    print('file {}, {} / {}'.format(m, len(b), len(list_files)))

    # c = list(set(list_files) - set(b))
    # c = random.sample(c, k=min(3*len(b), len(c)))
    new_image2 = []
    for row in images:
        if row['file_name'] in list_files:
        # if row['id'] in b:
            copy_row = copy.deepcopy(row)
            new_image2.append(copy_row)


    new_razmetka = dict(annotations=new_annotation, images=new_image2,
                        categories=categories, info=info, licenses=licenses)

    with open(os.path.join('/home/neptun/PycharmProjects/datasets/coco/labelstrain', f'{m}.json'), 'w') as f:
        f.write(json.dumps(new_razmetka))

if __name__ == '__main__':
    list_files = [
        "000000237618.jpg",
        "000000302282.jpg",
        "000000373974.jpg",
        "000000252570.jpg",
        "000000201630.jpg",
        "000000355225.jpg",
        "000000441576.jpg",
        "000000572016.jpg",
        "000000088241.jpg",
        "000000438126.jpg",
        "000000391145.jpg",
        "000000241643.jpg",
        "000000178658.jpg",
        "000000022271.jpg",
        "000000150704.jpg",
        "000000526927.jpg",
        "000000327988.jpg",
        "000000101752.jpg",
        "000000064697.jpg",
        "000000062865.jpg",
        "000000022482.jpg",
        "000000266517.jpg",]
    write_json(list_files, 3)
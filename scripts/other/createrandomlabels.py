import os
import random
import json
import copy


def make_file(N):
    current_label = 17  #cat
    # N = 1000
    path_to_dataset = '/home/neptun/PycharmProjects/datasets/coco'

    with open(os.path.join(path_to_dataset, 'instances_train2017.json')) as f:
        razmetka = json.load(f)

    categories = razmetka['categories']
    annotations = razmetka['annotations']
    images = razmetka['images']
    info = razmetka['info']
    licenses = razmetka['licenses']

    all_photo = []
    for row in annotations:
        all_photo.append(row['image_id'])
    all_photo = list(set(all_photo))

    if N != -1:
        all_photo = random.sample(all_photo, k=N)

    new_annotation = []
    for row in annotations:
        if row['category_id'] == current_label and row['image_id'] in all_photo:
            copy_row = copy.deepcopy(row)
            copy_row['segmentation'] = []
            new_annotation.append(copy_row)

    images_ids = []
    for row in new_annotation:
        images_ids.append(row['image_id'])
    images_ids = list(set(images_ids))

    new_image = []
    a = []
    for row in images:
        if row['id'] in images_ids:
            a.append(row['id'])
            copy_row = copy.deepcopy(row)
            new_image.append(copy_row)

    end_annot = []
    for row in new_annotation:
        if row['image_id'] in a:
            end_annot.append(copy.deepcopy(row))

    new_razmetka = dict(annotations=end_annot, images=new_image,
                        categories=categories, info=info, licenses=licenses)

    with open(os.path.join(path_to_dataset, 'labelstrain', 'first.json'), 'w') as f:
        f.write(json.dumps(new_razmetka))

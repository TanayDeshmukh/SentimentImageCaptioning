import json

karpathy_json_path = '/netscratch/deshmukh/train_valid_test_splits/dataset_flickr8k.json'

with open(karpathy_json_path, 'r') as f:
    data = json.load(f)

images = data['images']
test_image_paths = []

for i in range(len(images)):
    image = images[i]
    if image['split'] in {'test'}:
        test_image_paths.append(image['filename'])

print(test_image_paths[:10])
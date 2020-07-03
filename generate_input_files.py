from utils import create_input_files

if __name__ == '__main__':
    create_input_files(dataset = 'flickr8k', 
                        karpathy_json_path = '../../Datasets/train_valid_test_splits/dataset_flickr8k.json', 
                        image_folder = '../../Datasets/Flickr8K/Flicker8k_Dataset/', 
                        captions_per_image = 5, 
                        min_word_freq = 5, 
                        ouput_folder = './data') # This will be in /netscratch/deshmukh
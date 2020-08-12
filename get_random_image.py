import os, random
import tarfile

def get_random_image(imagenet_dir, synset=None, image_index=None):

  if synset is None:
    zip_list = os.listdir(imagenet_dir)
    synset_tar = random.sample(zip_list, 1)[0]
    synset = synset_tar.split('.')[0]

  else:
    synset_tar = synset+'.tar'

  #print(f'synset tar ----> {synset_tar}')
  #print(synset_tar_done)

  if synset_tar not in synset_tar_done:

    #print(synset_tar)
    #print('not in')

    file = os.path.join('/content/Images', synset)
    os.makedirs(file, exist_ok=True)
    #!tar -C  $file -xvf $synset_tar
    my_tar = tarfile.open(synset_tar)
    my_tar.extractall(file)
    my_tar.close()
    synset_tar_done.append(synset_tar)

  #print(synset_tar_done)

  if image_index is None:
    images_list = os.listdir(os.path.join('/content/Images', synset))
    image = random.sample(images_list, 1)[0]

  else:
    images_list = os.listdir(os.path.join('/content/Images', synset))
    images_list = sorted(images_list)
    image = images_list[image_index]
    #image = os.path.join(synset.split('.')[0], '_', image_index, '.JPEG')

  return image

synset_tar_done = []
imagenet_dir = '/content/drive/My Drive/Interpretability Research/imagenet_val/'

#%cd $imagenet_dir
os.chdir(imagenet_dir)
os.makedirs('/content/Images', exist_ok = True)


## Both synset and image random ##

image = get_random_image(imagenet_dir, synset=None, image_index=None)
print(f'image ----> {image}')


## Both image index and synset given ##
image_index = 10
synset = 'n02002724'
image = get_random_image(imagenet_dir, synset=synset, image_index=10)
print(f'image ----> {image}')

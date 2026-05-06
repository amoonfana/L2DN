import os
from torchvision.datasets import ImageFolder
import pickle

os.chdir("../") # Change the working directory to its parent directory, where train.py is located there

def gen_img_list(img_folder, list_name):
    dataset = ImageFolder("../Data/" + img_folder)
    imgs = dataset.imgs
    
    a = open("../Data/" + list_name + ".pickle", "wb")
    pickle.dump(imgs, a)
    a.close()
    
    print(len(imgs))
    print(imgs[-1][1]+1)
    
# ratio should be 0~9
def gen_webface42m_list(img_folder, list_name, ratio):
    dataset = ImageFolder("../Data/" + img_folder)

    root = dataset.root
    wf42m = dataset.imgs

    # Define how many folders (0~9) are included in the image list
    i_zip = len(root) + 1

    i_end = 0
    for img, _ in wf42m:
        if int(img[i_zip]) < ratio:
            i_end += 1
        else:
            break

    imgs = wf42m[:i_end]
        
    print(len(imgs))
    print(imgs[-1][1]+1)

    file = open("../Data/" + list_name + ".pickle", "wb")
    pickle.dump(imgs, file)
    file.close()
    
def modify_folder_in_img_list(addr_old, addr_new, folder_old, folder_new):
    file_old = open(addr_old, "rb")
    imgs = pickle.load(file_old)
    file_old.close()

    for i in range(len(imgs)):
        imgs[i] = list(imgs[i])
        imgs[i][0] = imgs[i][0].replace(folder_old, folder_new)
        imgs[i] = tuple(imgs[i])
        
    file_new = open(addr_new, "wb")
    pickle.dump(imgs, file_new)
    file_new.close()
    
# gen_img_list("ms1mv3", "mv31")
gen_webface42m_list("WebFace260M", "wf12m2", 3)
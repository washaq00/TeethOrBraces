# import argparse
import requests
# from torchvision.transforms import Lambda
from pathlib import Path
# from configs.config_utils import CONFIG
import os
from CNN import TinyVgg
from torch import nn
import torch
from CustomDataset import find_classes, CustomImageDataset
from myfunctions import print_image, training_loop
from torchvision import transforms
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn, plot_loss_curves


# def parse_args():
#     """User-friendly command lines"""
#     parser = argparse.ArgumentParser('Total 3D Understanding.')
#     parser.add_argument('--train', action='store_true')
#     parser.add_argument('--test', action='store_true')
#     return parser
torch.cuda.manual_seed(42)


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.is_available())
    # if Path("helper_functions.py").is_file():
    #   print("helper_functions.py already exists, skipping download")
    # else:
    #   request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    #   with open("helper_functions.py", "wb") as f:
    #     f.write(request.content)

    BATCH_SIZE = 32
    NUM_WORKERS = 4

    #implement Data Augmentation and Image Resizing
    train_transform = transforms.Compose([
                        transforms.Resize(size=(256, 256)),
                        transforms.TrivialAugmentWide(num_magnitude_bins= 31),
                        transforms.ToTensor()])

    test_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()])

    train_dir = os.path.join(os.getcwd(),"data/train/")
    # print(find_classes(train_dir))
    test_dir = os.path.join(os.getcwd(),"data/test/")

    train_data = CustomImageDataset(img_dir=train_dir,  # target folder of images
                                    transform=train_transform)  # transforms to perform on labels (if necessary)

    test_data = CustomImageDataset(img_dir=test_dir,
                                   transform=test_transform)

    print(f"test data {train_data}")
    class_names = train_data.classes
    # print(class_names)
    # img, label = train_data[0][0], train_data[0][1]
    # print_image(img)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  num_workers=NUM_WORKERS,
                                  shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False)  # don't usually need to shuffle testing data

    # img, label = next(iter(train_dataloader))
    # print(img, img.shape)
    # print_image(img[0])

    torch.manual_seed(42)

    model_0 = TinyVgg(input_shape=3,
                      hidden_units=10,
                      output_shape=len(train_data.classes)).to(device)

    # print(model_0)
    # model_0.eval()
    # imagee = img[1].unsqueeze(dim=0)
    # print(img[1].shape, imagee.shape)
    #
    # with torch.inference_mode():
    #     pred = model_0(imagee)
    #
    # print(pred)
    # print(torch.argmax(torch.softmax(pred, dim=1),dim=1))
    # print(label[1])

    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model_0_results = training_loop(model=model_0,
                                    train=train_dataloader,
                                    test=test_dataloader,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn,
                                    acc=accuracy_fn,
                                    epochs=50)

    plot_loss_curves(model_0_results)


if __name__ == '__main__':
    main()
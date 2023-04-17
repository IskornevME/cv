import torchvision
from IPython.display import display


def show_image(data, idx_to_label, mode, num_image_to_show=3, transformed=False):
    for i in range(num_image_to_show):
        if mode == "train":
            img = data[i][0]
            if transformed:
                print(img.shape)
                img = torchvision.transforms.ToPILImage()(img).convert("RGB")
            display(img)
            print(idx_to_label[data[i][1].item()])
            print()
        else:
            img = data[i][0]
            img_name = data[i][1]
            if transformed:
                print(img.shape)
                img = torchvision.transforms.ToPILImage()(img).convert("RGB")
            display(img)
            print(img_name)
            print()

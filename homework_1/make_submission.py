import pandas as pd
from tqdm import tqdm


def generate_submission(test_dataloader, model, idx_to_label, name, device_name):
    idxs = []
    pred_sports = []
    for batch in tqdm(test_dataloader):
        x, image_id = batch
        x = x.to(device_name)
        idxs += image_id
        outputs = model(x)
        pred_labels = outputs.max(dim=1)[1]
        pred_sports += [idx_to_label[label.item()] for label in pred_labels]

    data = {"image_id": idxs, "label": pred_sports}
    res = pd.DataFrame(data)
    res.to_csv('submissions/' + name, index=False)

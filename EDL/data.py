from data_handler.dataloaders import load_data

dataloader_train, dataloader_val, dataloader_test = load_data()
ds = {
    "train": dataloader_train,
    "val": dataloader_val,
}

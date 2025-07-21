import os
import sys
import copy
import click
import wandb
import random

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from pandas import DataFrame
from shaphypetune import BoostRFE, BoostRFA
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("/localdisk1/PARK/park_vlm/Dataset")
sys.path.append("/localdisk1/PARK/park_vlm/VideoEmbeddings")
sys.path.append("/localdisk1/PARK/park_vlm/Utils")

from file_path_labels import *
from get_static_embeddings import *
from models import *
from calculate_performance_metrics import *

# Setup GPU support
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"Device: {device}")

def select(features:DataFrame, labels, **cfg):
    '''
    Rank the features based on a "base model", 
    return top-n features where n is a hyper-parameter
    '''
    methods = { "BoostRFE":BoostRFE, "BoostRFA":BoostRFA}

    SELECTOR = methods[cfg["selector"]]

    base = XGBRegressor() if cfg["selector_base"] == "XGB" else LGBMRegressor()
    
    selector = SELECTOR(base)
    selector.fit(features, labels)

    sorts = selector.ranking_.argsort()
    selected_features = features.columns[sorts][:cfg["n"]]
    return selected_features

def construct_dataset_df(config, null_features=['blink_intensity_variability', 'lips_puller_intensity_variability', 'lips_tightening_intensity_variability', 'lips_parting_intensity_variability', 'jaw_drop_intensity_variability', 'inner_brow_raiser_intensity_variability', 'upper_lid_raiser_intensity_variability', 'cheek_raiser_intensity_variability', 'lid_tightener_intensity_variability']):
    '''
    Load surrogate features from full video, 
    and attach them to PD/Non-PD labels.
    Keep the unique ids
    '''
    # get the locations and PD labels for all files that exist in metadata
    dataset = get_file_paths_and_labels(dataset_name=config["dataset_name"])
    print(f"Size of training set: {len(dataset['train'])} videos, {len(dataset['train']['pid'].unique())} participants")
    print(f"Size of validation set: {len(dataset['dev'])} videos, {len(dataset['dev']['pid'].unique())} participants")
    print(f"Size of test set: {len(dataset['test'])} videos, {len(dataset['test']['pid'].unique())} participants")

    # combine all landmark features
    landmark_features_path = "/localdisk1/PARK/park_vlm/SurrogateFeatures/LandmarkFeatures"
    df_landmark = None
    for filename in os.listdir(landmark_features_path):
        if filename[-4:] == ".csv":
            df_temp = pd.read_csv(os.path.join(landmark_features_path, filename))
            if df_landmark is None:
                df_landmark = df_temp
            else:
                assert len(df_landmark.columns)==len(df_temp.columns)
                df_landmark = pd.concat([df_landmark, df_temp])
            
    print(f"Size of df_landmark: {len(df_landmark)}")
    # for x in df_landmark.columns:
    #     print(x)

    # combine all openface features
    openface_features_path = "/localdisk1/PARK/park_vlm/SurrogateFeatures/OpenfaceFeatures"
    df_openface = None
    for filename in os.listdir(openface_features_path):
        if filename[-4:] == ".csv":
            df_temp = pd.read_csv(os.path.join(openface_features_path, filename))
            if df_openface is None:
                df_openface = df_temp
            else:
                assert len(df_openface.columns)==len(df_temp.columns)
                df_openface = pd.concat([df_openface, df_temp])
            
    print(f"Size of df_openface: {len(df_openface)}")

    df_openface["filename"] = df_openface["filename"].astype(str)
    df_landmark["filename"] = df_landmark["filename"].astype(str)
    df_landmark["filename"] = df_landmark["filename"].apply(lambda x: x+"mp4")
    df_openface["filename"] = df_openface["filename"].apply(lambda x: x+".mp4")

    df_surrogate_features = df_openface.set_index("filename").join(df_landmark.set_index("filename"), on="filename", how="inner")
    df_surrogate_features = df_surrogate_features.reset_index()

    print(f"{len(df_surrogate_features.columns)} features, {len(df_surrogate_features)} data")
    
    # Identify null features
    # size_of_all_data = len(df_surrogate_features)
    # for feature in df_surrogate_features.columns:
    #     size_of_data_with_feature = len(df_surrogate_features.dropna(subset=[feature]))
    #     not_null_ratio = (size_of_data_with_feature/size_of_all_data)*100
    #     if not_null_ratio<100.0:
    #         print(f"feature: {feature}, not-null ratio: {not_null_ratio}")

    # remove null features
    df_surrogate_features = df_surrogate_features.drop(columns=null_features)
    
    # further remove rows with null values
    df_surrogate_features = df_surrogate_features.dropna()

    '''
    Drop metadata columns to focus on features
    '''
    df_features = df_surrogate_features.drop(columns=['filename'])

    '''
    Drop columns (if set true) if it is correlated with another one with PCC>thr
    '''
    if config["drop_correlated_features"]=="yes":
        corr_matrix = df_features.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        for i in iters:
            for j in range(i+1):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                if val >= config["corr_thr"]:
                    drop_cols.append(col.values[0])

        drops = set(drop_cols)

        # print("Dropped surrogate features")
        # for x in list(drops):
        #     print(x)
        # print("##"*10)
        
        # Drop features from both the main and the feature dataframe
        df_surrogate_features.drop(drops, axis=1, inplace=True)
        df_features.drop(drops, axis=1, inplace=True)

    print(f"Size of surrogate features: {len(df_surrogate_features)}")
    print(f"Number of surrogate features: {len(df_surrogate_features.columns)-1}")
    feature_names = df_surrogate_features.columns[1:]

    # combined embedding and label columns into a single dataframe
    for fold in ["train", "dev", "test"]:
        dataset[fold] = dataset[fold].set_index("file_name").join(df_surrogate_features.set_index("filename"), on="file_name", how="inner")
        dataset[fold] = dataset[fold].reset_index()
        print(f"Number of {fold} files with embedding: {len(dataset[fold])}")
    
    # dictionary of three dataframes: 
    # dataset["train"], dataset["dev"], and dataset["test"]
    return dataset, feature_names

class TensorDataset(Dataset):
    '''
    Standard dataloader for this specific single-view embeddings.
    '''
    def __init__(self, df, feature_names):
        self.ids = df["unique_id"]
        self.labels = df["label"]
        self.features = df[feature_names]
    
    def __getitem__(self, index):
        uid = self.ids.iloc[index]
        x = np.asarray(self.features.iloc[index])
        y = self.labels.iloc[index]
        return uid, x, y
    
    def __len__(self):
        return len(self.labels)
    
def evaluate(model, data_loader, criterion):
    '''
    Evaluation loop (model is not updated)
    Returns the loss and performance metrics
    '''
    model.eval()

    loss_total = 0
    n_total = 0
    all_preds = []
    all_labels = []

    for idx, batch in enumerate(data_loader):
        ids, features, labels = batch
        n_total += len(labels)
        all_labels.extend(labels)

        labels = labels.float().to(device)
        features = features.float().to(device)

        with torch.no_grad():
            predicted_probs = model(features)
            l = criterion(predicted_probs.reshape(-1), labels)
            loss_total += l.item()*len(labels)
            all_preds.extend(predicted_probs.to('cpu').numpy())

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = loss_total/n_total
    return metrics

def training_loop(train_loader, dev_loader, model, optimizer, scheduler, criterion, all_configs):
    '''
    Control the training process.
    Return the best model and performance metrics on training and validation sets.
    '''
    best_model = copy.deepcopy(model)
    best_dev_loss = np.finfo('float32').max
    loss_vs_iterations = [] # to plot the trend of training loss

    for epoch in tqdm(range(all_configs["num_epochs"])):
        # track training loss for each epoch
        training_loss = 0
        n_total = 0
        for idx, batch in enumerate(train_loader):
            ids, features, labels = batch
            n_total += len(labels)
            
            labels = labels.float().to(device)  #(n, )
            
            features = features.float().to(device)      #(n, d)
            
            predicted_probs = model(features)   #(n, 1)
            
            l = criterion(predicted_probs.reshape(-1), labels)
            
            # propagate gradients
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            training_loss += l.item()*len(labels)
            loss_vs_iterations.append(l.item())

        training_loss = training_loss/n_total   # average training loss
        print(training_loss)
        
        if all_configs["enable_wandb"]:
            wandb.log({"train_loss": training_loss})
        
        # evaluate performance on the validation set
        dev_metrics = evaluate(model, dev_loader, criterion)
        print(f"After epoch {epoch}, training_loss = {training_loss}, validation loss = {dev_metrics['loss']}")
        
        if all_configs["use_scheduler"]=="yes":
            if all_configs['scheduler']=='step':
                scheduler.step()
            else:
                scheduler.step(dev_metrics["loss"])

        if all_configs["detailed_logs"]:
            print("Metrics on vaidation set:")
            print(dev_metrics)

        # update the best model if this epoch improved validation performance (loss)
        if dev_metrics["loss"] < best_dev_loss:
            best_dev_loss = dev_metrics["loss"]
            best_model = copy.deepcopy(model)
            if all_configs["detailed_logs"]:
                print("Model updated")

        print("---"*3)
        # end of one epoch

    # end of training
    # evaluate final performance on the training and validation set
    training_metrics = evaluate(best_model, train_loader, criterion)
    dev_metrics = evaluate(best_model, dev_loader, criterion)
    
    # save training loss curve
    if all_configs["detailed_logs"]:
        plt.plot(np.arange(len(loss_vs_iterations)), loss_vs_iterations)
        plt.savefig("training_loss.png", dpi=300)

    return best_model, training_metrics, dev_metrics

@click.command()
@click.option("--num_epochs", default=10)
@click.option("--batch_size", default=256)
@click.option("--hidden_dim", default=512)
@click.option("--drop_prob", default=0.5)
@click.option("--optimizer",default="AdamW",help="Options: SGD, AdamW")
@click.option("--learning_rate", default=0.001, help="Learning rate for classifier")
@click.option("--momentum", default=0.9)
@click.option("--use_scheduler", default='no',help="Options: yes, no")
@click.option("--scheduler", default='step',help="Options: step, reduce")
@click.option("--step_size", default=11)
@click.option("--gamma", default=0.8808588244592819)
@click.option("--patience", default=3)
@click.option("--detailed_logs", default=False)
@click.option("--seed", default=42)
@click.option("--enable_wandb", default=True)
@click.option("--drop_correlated_features", default='yes')
@click.option("--corr_thr", default=0.85)
@click.option("--selector", default="BoostRFE")
@click.option("--selector_base", default="XGB")
@click.option("--n", default=20)
@click.option("--use_feature_selection", default="yes")
@click.option("--dataset_name", default="finger_tapping")
def main(**cfg):
    # need to setup wandb and hyper-parameter tuning
    ENABLE_WANDB = cfg["enable_wandb"]
    if ENABLE_WANDB:
        wandb.init(project="park_vlm_v0", config=cfg)
    '''
    Ensure reproducibility of randomness
    '''
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"]) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    # these are configs for model training
    all_configs = cfg

    # these are configs for embedding extraction
    # datasets: all_tasks, facial_expressions, free_flow_speech
    config = {
        "dataset_name": cfg["dataset_name"],
        "drop_correlated_features": cfg["drop_correlated_features"],
        "corr_thr": cfg["corr_thr"]
    }

    # setup dataset, data loader, model, optimizer, scheduler, loss function
    dataset_df, feature_names = construct_dataset_df(config)
    
    if cfg["use_feature_selection"]=="yes":
        metadata_columns = ['file_name', 'pid', 'task', 'protocol', 'label', 'file_path', 'unique_id']
        selected_features = select(dataset_df["train"].drop(columns=metadata_columns), list(dataset_df["train"]["label"]), **all_configs)
        print(selected_features)
        # print(f"Number of selected features: {len(selected_features)}")

        drop_columns = [x for x in dataset_df["train"].columns if ((x not in selected_features) and (x not in metadata_columns))]
    
        for fold in ["train", "dev", "test"]:
            dataset_df[fold] = dataset_df[fold].drop(columns=drop_columns)
        
        feature_names = selected_features

    train_dataset = TensorDataset(df=dataset_df["train"], feature_names=feature_names)
    dev_dataset = TensorDataset(df=dataset_df["dev"], feature_names=feature_names)
    test_dataset = TensorDataset(df=dataset_df["test"], feature_names=feature_names)
    # print(len(train_dataset))
    # uid, x, y = train_dataset[0]
    # print(x.shape)
    # print(uid, y)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=all_configs["batch_size"], shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=all_configs["batch_size"])
    test_loader = DataLoader(dataset=test_dataset, batch_size=all_configs["batch_size"])
    
    _, x, _ = train_dataset[0]
    n_features = x.shape[0]
    model = LinearProbe(n_features=n_features, hidden_dim=all_configs["hidden_dim"], drop_prob=all_configs["drop_prob"])
    model.to(device)
    
    criterion = nn.BCELoss()

    if all_configs["optimizer"]=="AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=all_configs['learning_rate'])
    elif all_configs["optimizer"]=="SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=all_configs['learning_rate'], momentum=all_configs['momentum'])
    elif all_configs["optimizer"]=="RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=all_configs['learning_rate'], momentum=all_configs['momentum'])
    else:
        raise ValueError("Invalid optimizer")
    
    scheduler = None
    if all_configs["use_scheduler"]=="yes":
        if all_configs['scheduler']=="step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=all_configs['step_size'], gamma=all_configs['gamma'])
        elif all_configs['scheduler']=="reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=all_configs['gamma'], patience = all_configs['patience'])
        else:
            raise ValueError("Invalid scheduler")

    # train the model    
    trained_model, training_metrics, dev_metrics = training_loop(
        train_loader=train_loader, dev_loader=dev_loader, model=model, 
        optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        all_configs=all_configs)
    
    wandb_logs = {}
    print("="*10)
    print("Training metrics")
    print(training_metrics)
    wandb_logs["train_loss"] = training_metrics["loss"]
    print("Validation metrics")
    print(dev_metrics)
    wandb_logs["dev_loss"] = dev_metrics["loss"]
    wandb_logs["dev_auroc"] = dev_metrics["auroc"]
    wandb_logs["dev_accuracy"] = dev_metrics["accuracy"]
    wandb_logs["dev_f1_score"] = dev_metrics["f1_score"]
    
    # evaluate on the test set
    test_metrics = evaluate(model=trained_model, data_loader=test_loader, criterion=criterion)
    print("Test metrics")
    print(test_metrics)
    wandb_logs["test_accuracy"] = test_metrics["accuracy"]
    wandb_logs["test_precision"] = test_metrics["precision"]
    wandb_logs["test_recall"] = test_metrics["recall"]
    wandb_logs["test_auroc"] = test_metrics["auroc"]
    wandb_logs["test_f1_score"] = test_metrics["f1_score"]
    if ENABLE_WANDB:
        wandb.log(wandb_logs)
    # save the model
    # after we find out the best configuration

if __name__ == "__main__":
    main()
        
    

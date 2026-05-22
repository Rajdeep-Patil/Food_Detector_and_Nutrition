import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import sys

from src.constants import BATCH_SIZE,EPOCHS,LEARNING_RATE,NUM_CLASSES,MODEL_PATH,CLASSES_PATH

from src.utils.helper import save_model, save_object, train_one_epoch, evaluate
from src.logger import logger
from src.exception import CustomException


class ModelTrainer:
    def initiate_model_training(self,train_dataset,test_dataset,class_names=None):

        logger.info("Starting AlexNet Training...")

        try:

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)

            val_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

            alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

            for param in alexnet.features.parameters():
                param.requires_grad = False

            alexnet.classifier = nn.Sequential(

                nn.Dropout(0.5),

                nn.Linear(256*6*6,1024),

                nn.ReLU(),

                nn.Dropout(0.5),

                nn.Linear(1024,512),

                nn.ReLU(),

                nn.Linear(512,NUM_CLASSES)
            )

            alexnet = alexnet.to(device)

            criterion = nn.BCEWithLogitsLoss()

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, alexnet.parameters()),lr=LEARNING_RATE,weight_decay=1e-4)

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=8,gamma=0.5)

            print(f"{'Epoch':<8}"f"{'Train Loss':<18}"f"{'Train Acc':<18}"f"{'Val Loss':<18}"f"{'Val Acc':<18}"f"{'LR':<10}")

            print("-"*90)

            for epoch in range(EPOCHS):

                avg_train_loss, train_acc = train_one_epoch(alexnet,train_loader,criterion,optimizer,device)

                avg_val_loss, val_acc = evaluate(alexnet,val_loader,criterion,device)

                scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]

                print(f"{epoch+1:<8}"f"{avg_train_loss:<18.4f}"f"{train_acc:<18.4f}"f"{avg_val_loss:<18.4f}"f"{val_acc:<18.4f}"f"{current_lr:<10.6f}")

                logger.info(f"Epoch [{epoch+1}/{EPOCHS}] | "f"Train Loss: {avg_train_loss:.4f} | "f"Train Acc: {train_acc:.4f} | "f"Val Loss: {avg_val_loss:.4f} | "f"Val Acc: {val_acc:.4f} | "f"LR: {current_lr:.6f}")

            save_model(MODEL_PATH,alexnet)

            if class_names:

                save_object(CLASSES_PATH,class_names)

            return alexnet, device

        except Exception as e:
            raise CustomException(e,sys)
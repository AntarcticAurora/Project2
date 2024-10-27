from ultralytics import YOLO
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of pretrained model', type=str, required=True)
    parser.add_argument('--data_config_path', help='path of dataset config', type=str,required=True)
    parser.add_argument('--img_size', help='image pixel size', type=int, required=True)
    parser.add_argument('-n','--name', help='name of this training', type=str, default="yolo_train")
    parser.add_argument('-b','--batch_size', help='batch size', type=int, default=64)
    parser.add_argument('-e', '--epoch', help='epoch number', type=int, default=10)
    args = parser.parse_args()
    device = torch.device("cpu")
    # load the model
    model = YOLO(args.model_path)
    model.to(device)
    print("Using device: ", device)
    # training
    model.train(
        data=args.data_config_path,
        imgsz = args.img_size,
        epochs = args.epoch,
        batch=args.batch_size,
        name = args.name,
    )


if __name__ == "__main__":
    main()

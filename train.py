import argparse
from utils.train_utils import apply_transform, prepare_dataloaders, prepare_objectives, prepare_models
from torchsummary import summary
import yaml
import torch
from trainer import train_and_evaluate
from utils.helper import get_args_parser
from preprocess_data import preprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Temperature prediction training script", parents=[get_args_parser()])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)

    data = preprocess(config)

    train_transform, valid_transform = apply_transform(config)
    training_loader, validation_loader = prepare_dataloaders(
        config, data, train_transform, valid_transform)
    
    model = prepare_models(config)
    model.to(config['device'])

    criterion, optimizer, scheduler = prepare_objectives(config, model)

    if config['continue_training']:
        model.load_state_dict(torch.load(
            config['trained_weights'])['model_state_dict'])
        epoch = torch.load(config['trained_weights'])['epoch']
    else:
        epoch = 0

    train_and_evaluate(config, training_loader, validation_loader,
                       model, criterion, optimizer, scheduler, epoch)

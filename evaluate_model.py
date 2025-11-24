import os
import random
import logging
import argparse

import torch
import numpy as np

from src.cnn import *
from src.data_augmentations import *
from src.eval.evaluate import count_trainable_parameters, eval_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Enforce deterministic behavior
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DL WS25/26 Competition')

    parser.add_argument('-m', '--model',
                        default='evaluation_model',
                        help='Name of the Model class present in cnn.py (Eg: SampleModel)',
                        type=str)

    parser.add_argument('-p', '--saved-model-file',
                        default='sample_model',
                        help='Name of file inside models directory which contains the saved weights of the trained '
                             'model',
                        type=str)

    parser.add_argument('-D', '--test-data-dir',
                        default=os.path.join(os.getcwd(), 'dataset', 'test'),
                        help='Path to folder with the test data to evaluate the model on.'
                        + 'The organizers will populate the test folder with the unseen dataset to evaluate your model.'
                        )

    parser.add_argument('-d', '--data-augmentations',
                        default='evaluation_augmentation',
                        help='Data augmentation to apply to data before passing it to the model. '
                        + 'Must be available in data_augmentations.py')

    parser.add_argument('-n', '--num-params',
                        action='store_true',
                        help='Print the number of trainable parameters in the model',)

    parser.add_argument('-v', '--verbose',
                        default='INFO',
                        choices=['INFO', 'DEBUG'],
                        help='verbosity')

    args, unknowns = parser.parse_known_args()

    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    model_class = eval(args.model)

    if args.num_params:
        num_params = count_trainable_parameters(model_class())
        print(f"Number of trainable parameters in the model: {num_params}")
        with open("n_params.txt", "w") as f:
            f.write(str(num_params))
        exit(0)

    else:
        eval_model(
            model=model_class(),
            saved_model_file=args.saved_model_file,
            test_data_dir=args.test_data_dir,
            data_augmentations=eval(args.data_augmentations),
        )

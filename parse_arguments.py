"""This module contains the CLI arguments parser for the defect detection model training."""

import argparse


def get_training_parser(parser: argparse.ArgumentParser = None):
    """Parse CLI arguments to train defect detection model."""
    if parser is None:
        parser = argparse.ArgumentParser()

    data_inputs = parser.add_argument_group("data_inputs")
    data_inputs.add_argument(
        "--train_data",
        type=str,
        required=False,
        nargs="*",
        help="Path to the data root.",
    )
    data_inputs.add_argument(
        "--train_annots",
        type=str,
        required=False,
        nargs="*",
        help="Path to the annotations.",
    )
    data_inputs.add_argument(
        "--valid_data",
        type=str,
        required=False,
        nargs="*",
        help="Path to the data root.",
    )
    data_inputs.add_argument(
        "--valid_annots",
        type=str,
        required=False,
        nargs="*",
        help="Path to the annotations.",
    )
    data_inputs.add_argument(
        "--test_data",
        type=str,
        required=False,
        nargs="+",
        help="Path to the data root.",
    )
    data_inputs.add_argument(
        "--test_annots",
        type=str,
        required=False,
        nargs="+",
        help="Path to the annotations.",
    )

    pl_trainer_settings = parser.add_argument_group("pl_trainer_settings")
    pl_trainer_settings.add_argument(
        "--max_epochs",
        type=int,
        default=45,
        help="Number of epochs for training. Default is 10.",
    )
    pl_trainer_settings.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=0,
        help="Device(s) to use for training.",
    )
    pl_trainer_settings.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Training strategy.",
    )
    pl_trainer_settings.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["bf16", "16-mixed", "32"],
        help="Precision for training.",
    )
    pl_trainer_settings.add_argument(
        "--sync_batchnorm",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Synchronize batch normalization layers.",
    )
    pl_trainer_settings.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients. Default is 1.",
    )
    pl_trainer_settings.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=5,
        help="Check validation every n epochs.",
    )
    pl_trainer_settings.add_argument(
        "--num_nodes", type=int, default=1, help="Number of nodes for training."
    )
    pl_trainer_settings.add_argument(
        "--fast_dev_run", type=str, default="False", choices=["True", "False"]
    )
    pl_trainer_settings.add_argument(
        "--log_every_n_steps",
        type=int,
        default=50,
        help="Log every n steps.",
    )
    pl_trainer_settings.add_argument(
        "--enable_progress_bar",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Enable progress bar.",
    )

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for inference. Default is 256.",
    )
    optimization.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate for training. Default is 1e-3.",
    )
    optimization.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for training. Default is 0.9.",
    )
    optimization.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for training. Default is 1e-4.",
    )
    optimization.add_argument(
        "--warmup_start_lr",
        type=float,
        default=3e-5,
        help="Warmup start learning rate.",
    )
    optimization.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of warmup epochs.",
    )
    optimization.add_argument(
        "--exclude_bias_and_norm",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Exclude bias and normalization layers from weight decay.",
    )
    optimization.add_argument(
        "--eta_min",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine annealing.",
    )
    optimization.add_argument(
        "--scheduler_interval",
        type=str,
        default="step",
        choices=["epoch", "step"],
        help="Scheduler interval for learning rate adjustment.",
    )
    optimization.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["sgd", "lars", "adam", "adamw"],
        help="Optimizer for training.",
    )
    optimization.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "multistep", "none"],
        help="Scheduler for learning rate adjustment.",
    )
    optimization.add_argument(
        "--lr_steps",
        type=int,
        nargs="+",
        default=[1, 8, 20],
        help="Steps for the 'multistep' learning rate scheduler.",
    )
    optimization.add_argument(
        "--channels_last",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Use channels last format for training.",
    )
    optimization.add_argument(
        "--detach_backbone",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Detach backbone from the model. Only used in when fine-tuning a model.",
    )

    data_module_settings = parser.add_argument_group("data_module_settings")
    data_module_settings.add_argument(
        "--sampling_strategy",
        type=str,
        default="random",
        choices=["random", "binary-weighted"],
        help="Sampling strategy to use for training.",
    )
    data_module_settings.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading. Default is 4.",
    )
    data_module_settings.add_argument(
        "--effective_beta",
        type=float,
        default=0.9999,
        help="Beta value to compute effective weigthts.",
    )
    data_module_settings.add_argument(
        "--use_f2ciw_pos_weights",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Use F2CIW positive weights for training.",
    )
    data_module_settings.add_argument(
        "--imbalance_dataset",
        type=str,
        default="False",
        choices=["True", "False"],
    )
    data_module_settings.add_argument(
        "--defect_proportion",
        type=float,
        default=0.4,
        help="Imbalance level for the dataset.",
    )
    data_module_settings.add_argument(
        "--use_sample",
        type=str,
        default="False",
        choices=["True", "False"],
    )
    data_module_settings.add_argument(
        "--sample_percentage",
        type=float,
        default=0.25,
        help="Percentage of data to sample.",
    )

    model_settings = parser.add_argument_group("model_settings")
    model_settings.add_argument(
        "--task_type",
        type=str,
        default="binary",
        choices=["binary", "multi"],
        help="Type of classification task to perform.",
    )
    model_settings.add_argument(
        "--model_type",
        type=str,
        default="resnet18",
        choices=[
            "xie",
            "dino",
            "convnext-tiny",
            "convnext-small",
            "convnext-base",
            "convnext-large",
            "swin",
            "efficientnet",
            "resnet18",
            "resnet50",
            "vit-tiny",
            "vit-small",
            "mae-vit-small",
            "mae-vit-tiny",
        ],
    )
    model_settings.add_argument(
        "--pretrained",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Use pretrained weights for the model.",
    )
    model_settings.add_argument(
        "--img_size",
        type=int,
        default=224,
        choices=[64, 128, 224, 384, 512],
        help="Image size for training. Default is 224.",
    )

    parser.add_argument(
        "--learning_strategy",
        type=str,
        default="supervised",
        choices=["supervised", "ssl"],
    )
    parser.add_argument(
        "--only_log_on_epoch_end",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Log only on epoch end.",
    )

    load_model_settings = parser.add_argument_group("load_model_settings")
    load_model_settings.add_argument(
        "--load_model",
        type=str,
        default="False",
        choices=["False", "True"],
    )
    load_model_settings.add_argument(
        "--load_optimization_settings",
        type=str,
        default="False",
        choices=["False", "True"],
        help="Load optimization settings from a checkpoint.",
    )
    load_model_settings.add_argument(
        "--wandb_entity",
        type=str,
        help="Name of the wandb entity.",
    )
    load_model_settings.add_argument(
        "--wandb_project",
        type=str,
        help="Name of the wandb project.",
    )
    load_model_settings.add_argument(
        "--wandb_id",
        type=str,
        help="ID of the wandb run.",
    )
    load_model_settings.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to the model checkpoint.",
    )
    load_model_settings.add_argument(
        "--individual_components_path",
        type=str,
        default="individual_components/",
        help="Path to the individual components checkpoint.",
    )
    load_model_settings.add_argument(
        "--load_classifier",
        type=str,
        default="True",
        choices=["False", "True"],
        help="Whether or not to load the checkpoint classifier.",
    )

    momentum_encoder_settings = parser.add_argument_group("momentum_encoder")
    momentum_encoder_settings.add_argument(
        "--base_tau",
        type=float,
        default=0.996,
        help="Base tau for the momentum update.",
    )
    momentum_encoder_settings.add_argument(
        "--final_tau",
        type=float,
        default=0.999,
        help="Final tau for the momentum update.",
    )

    ssl_settings = parser.add_argument_group("ssl_settings")
    ssl_settings.add_argument(
        "--ssl_method",
        type=str,
        default="byol",
        choices=["byol", "simclr", "barlow_twins", "mae", "dino"],
        help="Self-supervised learning method.",
    )
    # DINO
    ssl_settings.add_argument(
        "--norm_prototyper",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Normalize the prototyper.",
    )
    ssl_settings.add_argument(
        "--clip_grad",
        type=str,
        default="True",
        choices=["True", "False"],
    )
    ssl_settings.add_argument(
        "--freeze_prototyper",
        type=int,
        default=1,
        help="Freeze the prototyper.",
    )
    ssl_settings.add_argument(
        "--student_temperature",
        type=float,
        default=0.1,
        help="Temperature for the student.",
    )
    ssl_settings.add_argument(
        "--teacher_temperature",
        type=float,
        default=0.07,
        help="Temperature for the teacher.",
    )
    ssl_settings.add_argument(
        "--warmup_teacher_temperature",
        type=float,
        default=0.04,
        help="Warmup temperature for the teacher.",
    )
    ssl_settings.add_argument(
        "--warmup_temperature_epochs",
        type=int,
        default=15,
        help="Warmup temperature epochs for the teacher.",
    )
    ssl_settings.add_argument(
        "--use_bn_in_head",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Use batch normalization in the head.",
    )
    ssl_settings.add_argument(
        "--num_prototypes",
        type=int,
        default=8192,
        help="Number of prototypes.",
    )
    # BYOL, DINO, SimCLR, Barlow Twins
    ssl_settings.add_argument(
        "--proj_hidden_dim",
        type=int,
        default=2048,
        help="Hidden dimension of the projector head.",
    )
    ssl_settings.add_argument(
        "--proj_output_dim",
        type=int,
        default=256,
        help="Output dimension of the projector head.",
    )
    # SimCLR
    ssl_settings.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for the softmax in contrastive loss.",
    )
    # Barlow Twins
    ssl_settings.add_argument(
        "--lamb",
        type=float,
        default=0.0051,
        help="Off-diagonal scaling factor for the cross-covariance matrix.",
    )
    ssl_settings.add_argument(
        "--scale_loss",
        type=float,
        default=0.024,
        help="Scaling factor of the loss.",
    )
    # BYOL
    ssl_settings.add_argument(
        "--predictor_hidden_dim",
        type=int,
        default=2048,
        help="Hidden dimension of the predictor head.",
    )
    ssl_settings.add_argument(
        "--normalize_projector",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Normalize the projector head.",
    )
    # MAE
    ssl_settings.add_argument(
        "--mask_ratio",
        type=float,
        default=0.75,
        help="percentage of image to mask.",
    )
    ssl_settings.add_argument(
        "--norm_pix_loss",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Normalize pixel loss.",
    )
    ssl_settings.add_argument(
        "--decoder_embed_dim",
        type=int,
        # TODO: look for default parameters in paper
        default=256,
        help="MAE's embedding dimension for the decoder.",
    )
    ssl_settings.add_argument(
        "--decoder_depth",
        type=int,
        default=3,
        help="MAE's depth for the decoder.",
    )
    ssl_settings.add_argument(
        "--decoder_num_heads",
        type=int,
        default=4,
        help="MAE's number of heads for the decoder.",
    )

    ssl_optimization = parser.add_argument_group("ssl_optimization")
    ssl_optimization.add_argument(
        "--linear_probe_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the linear probe.",
    )
    ssl_optimization.add_argument(
        "--linear_probe_momentum",
        type=float,
        default=0.9,
        help="Momentum for the linear probe.",
    )
    ssl_optimization.add_argument(
        "--linear_probe_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for the linear probe.",
    )
    ssl_optimization.add_argument(
        "--use_supervised_signal",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Use supervised signal for training.",
    )
    ssl_optimization.add_argument(
        "--num_large_crops",
        type=int,
        default=2,
        help="Number of large crops for training.",
    )
    ssl_optimization.add_argument(
        "--num_small_crops",
        type=int,
        default=0,
        help="Number of small crops for multicrop training.",
    )

    knn_eval = parser.add_argument_group("knn_eval")
    knn_eval.add_argument(
        "--use_knn_eval",
        type=str,
        default="False",
        choices=["True", "False"],
    )
    knn_eval.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of nearest neighbors for the kNN classifier.",
    )
    knn_eval.add_argument(
        "--T",
        type=float,
        default=0.07,
        help="Temperature for the kNN classifier.",
    )
    knn_eval.add_argument(
        "--distance_fx",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance function for the kNN classifier.",
    )
    knn_eval.add_argument(
        "--sample_train_size",
        type=float,
        default=1.0,
        help="Sample size for training.",
    )

    augmentations = parser.add_argument_group("augmentations")
    augmentations.add_argument(
        "--use_random_resized_crop",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Use random resized crop for training.",
    )
    augmentations.add_argument(
        "--min_scale",
        type=float,
        default=0.395,
        help="Minimum scale for random resized crop.",
    )
    augmentations.add_argument(
        "--max_scale",
        type=float,
        default=1.0,
        help="Maximum scale for random resized crop.",
    )
    augmentations.add_argument(
        "--use_color_jitter",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Use color jitter for training.",
    )
    augmentations.add_argument(
        "--brightness",
        type=float,
        default=0.275,
        help="Brightness factor for color jitter.",
    )
    augmentations.add_argument(
        "--contrast",
        type=float,
        default=0.275,
        help="Contrast factor for color jitter.",
    )
    augmentations.add_argument(
        "--saturation",
        type=float,
        default=0.275,
        help="Saturation factor for color jitter.",
    )
    augmentations.add_argument(
        "--hue",
        type=float,
        default=0.1375,
        help="Hue factor for color jitter.",
    )
    augmentations.add_argument(
        "--color_jitter_prob",
        type=float,
        default=1,
        help="Probability of applying color jitter.",
    )

    arbitrary_epoch_checkpoint = parser.add_argument_group("arbitrary_epoch_checkpoint")
    arbitrary_epoch_checkpoint.add_argument(
        "--percentages",
        type=float,
        nargs="+",
        default=[0.5],
        help="Percentages of epochs to save checkpoints.",
    )

    wandb_settings = parser.add_argument_group("wandb_settings")
    wandb_settings.add_argument(
        "--job_type",
        type=str,
        default="train",
        choices=["train", "inference"],
        help="Type of job to run.",
    )
    wandb_settings.add_argument(
        "--project",
        type=str,
        required=False,
        help="Name of the wandb project.",
    )
    wandb_settings.add_argument(
        "--entity",
        type=str,
        default="ssl-4-anomaly",
        required=False,
        help="Name of the wandb entity.",
    )
    wandb_settings.add_argument(
        "--group",
        type=str,
        required=False,
        help="Name of the wandb group.",
    )
    wandb_settings.add_argument(
        "--mode",
        type=str,
        default="online",
        choices=["offline", "online"],
    )
    wandb_settings.add_argument(
        "--id",
        type=str,
        help="ID of the wandb run.",
    )
    wandb_settings.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Tags for the wandb run.",
    )

    sanity_checks(parser)

    return parser


def sanity_checks(parser: argparse.ArgumentParser):
    """Perform sanity checks on parsed arguments."""
    args = parser.parse_args()

    if bool(args.train_data is None) != bool(args.train_annots is None):
        raise ValueError(
            "Either both training data and annotations must be provided or none of them."
        )

    if bool(args.valid_data is None) != bool(args.valid_annots is None):
        raise ValueError(
            "Either both validation data and annotations must be provided or none of them."
        )

    if args.train_data is not None and len(args.train_data) != len(args.train_annots):
        raise ValueError(
            "Number of training data folders must match number of annotation files."
        )

    if args.valid_data is not None and len(args.valid_data) != len(args.valid_annots):
        raise ValueError(
            "Number of validation data folders must match number of annotation files."
        )

    if args.job_type == "inference":
        if args.test_data is not None and len(args.test_data) != len(args.test_annots):
            raise ValueError(
                "Number of test data folders must match number of annotation files."
            )

    if args.img_size != 224 and args.model_type == "xie":
        raise ValueError(
            "Xie et al. model only supports image size of 224."
            " Please set image size to 224."
        )

    if args.sampling_strategy != "random" and args.task_type == "multi":
        raise ValueError(
            "Weighted sampling strategy is only supported for binary-class tasks."
        )

    if args.detach_backbone == "True" and args.learning_strategy != "supervised":
        raise ValueError(
            f"Cannot detach backbone when using 'learning_strategy' different from 'supervised'. Got '{args.learning_strategy}'."
        )

    if args.job_type == "inference" and args.load_model == "False":
        raise ValueError("Cannot perform inference without loading a model.")

    if args.load_model == "True" and args.wandb_id == "None":
        raise ValueError("Cannot load model without providing a `wandb_id`.")

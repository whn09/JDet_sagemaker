import argparse
import jittor as jt

import sys
sys.path.append('/opt/ml/code/JDet/python/')

from jdet.runner import Runner 
from jdet.config import init_cfg

def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="train",
        help="train,val,test",
        type=str,
    )

    parser.add_argument(
        "--no_cuda",
        action='store_true'
    )

    parser.add_argument(
        "--save_dir",
        default=".",
        type=str,
    )
    
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')  # os.environ['SM_MODEL_DIR']
    parser.add_argument('--output-dir', type=str, default='/opt/ml/output')  # os.environ['SM_MODEL_DIR']
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/train')  # os.environ['SM_CHANNEL_TRAINING']
    parser.add_argument('--test', type=str, default='/opt/ml/input/data/test')  # os.environ['SM_CHANNEL_TEST']
    
    args = parser.parse_args()

    if not args.no_cuda:
        jt.flags.use_cuda=1

    assert args.task in ["train","val","test","vis_test"],f"{args.task} not support, please choose [train,val,test,vis_test]"
    
    if args.config_file:
        init_cfg(args.config_file)

    runner = Runner()

    if args.task == "train":
        runner.run()
    elif args.task == "val":
        runner.val()
    elif args.task == "test":
        runner.test()
    elif args.task == "vis_test":
        runner.run_on_images(args.save_dir)

if __name__ == "__main__":
    main()
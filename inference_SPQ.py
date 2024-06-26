import torchvision.datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import *
from Retrieval import *
import os
from torchvision.transforms import InterpolationMode
from main_SPQ import Quantization_Head,CQCLoss
from PIL import Image

# python inference_SPQ.py  --in_weights 32_0.8719_checkpoint.pth  --image_x_path /Volumes/Sandi/Jewelry/R/CRE6981R01M14WA.jpg --image_y_path /Volumes/Sandi/Jewelry/R/CRE6985R01M14WA.jpg
def get_args_parser():
    parser = argparse.ArgumentParser('SPQ', add_help=False)

    parser.add_argument('--gpu_id', default="0", type=str, help="""Define GPU id.""")
    parser.add_argument('--image_x_path', default="", type=str, help="""Path of the dataset to be installed.""")
    parser.add_argument('--image_y_path', default="", type=str, help="""Path of the dataset to be installed.""")
    parser.add_argument('--input_size', default=210, type=int, help="""Input image size, default is set to CIFAR10.""")
    parser.add_argument('--device', default='cpu', type=str,choices = ['mps', 'cpu', 'cuda'], help="""device.""")

    parser.add_argument('--N_books', default=8, type=int, help="""The number of the codebooks.""")
    parser.add_argument('--N_words', default=16, type=int, help="""The number of the codewords. It should be a power of two.""")
    parser.add_argument('--L_word', default=16, type=int, help="""Dimensionality of the codeword.""")
    parser.add_argument('--soft_quantization_scale', default=5.0, type=float, help="""Soft-quantization scaling parameter.""")
    parser.add_argument('--contrastive_temperature', default=0.5, type=float, help="""Contrastive learning Temperature scaling parameter.""")
    
    parser.add_argument('--in_weights', default=".", type=str, help="""load checkpoints.""")
    # parser.add_argument('--output_dir', default=".", type=str, help="""Path to save logs and checkpoints.""")

    return parser

def infer(args):
    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        device = T.device('cuda')
    else:
        device = args.device
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = T.device(device)
    sz = args.input_size

    N_books = args.N_books
    N_words = args.N_words
    L_word = args.L_word
    tau_q = args.soft_quantization_scale

    N_bits = int(N_books * np.sqrt(N_words))
    print('\033[91m' + '%d'%N_bits +  '-bit to retrieval' + '\033[0m')

    img0 = Image.open(args.image_x_path)
    img1 = Image.open(args.image_x_path)

    # transform = transforms.ToTensor()
    tfs = transforms.Compose([
        transforms.Resize(size=(args.input_size,args.input_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    x = tfs(img0).unsqueeze_(0).to(device)
    y = tfs(img1).unsqueeze_(0).to(device)

    Q = Quantization_Head(N_words, N_books, L_word, tau_q)
    net = nn.Sequential(ResNet_Baseline(BasicBlock, [2, 2, 2, 2]), Q).to(device)
    net.load_state_dict(T.load(args.in_weights,map_location=T.device(args.device)))

    net.eval()
    with T.no_grad():
        Xa, Xb = net(x)
        Ya, Yb = net(y)

    gallery_codes = Indexing(Q.C, args.N_books, Xa).type(T.int)
    query_codes = Ya
    # res = T.argsort(pqDist_one(Q.C, N_books, gallery_codes, query_codes[0]))
    criterion = CQCLoss(device, 1, args.contrastive_temperature)
    loss = criterion(Xa,Ya, Xb, Yb)
    print(loss)
    res = pqDist_one(Q.C, N_books, gallery_codes, query_codes[0])
    print(res.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SPQ', parents=[get_args_parser()])
    args = parser.parse_args()
    infer(args)

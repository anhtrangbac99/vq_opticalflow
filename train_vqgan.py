import os
import argparse
from tqdm import tqdm
import torch
from taming.utils import load_data, instantiate_from_config
import OmegaConf
from taming.vqgan import VQGAN

def configure_optimizers(self, args):
    lr = args.learning_rate
    opt_vq = torch.optim.Adam(
        list(self.vqgan.encoder.parameters()) +
        list(self.vqgan.decoder.parameters()) +
        list(self.vqgan.codebook.parameters()) +
        list(self.vqgan.quant_conv.parameters()) +
        list(self.vqgan.post_quant_conv.parameters()),
        lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
    )
    opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

    return opt_vq, opt_disc

def prepare_training():
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

def train(vqgan,args):
    train_dataloader = load_data(args) #todo
    # train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size)
    optimizer_idx = args.optimizer_idx
    
    cmm_loss = 0
    # best_loss = 10^10
    # iter_per_epoch = np.ceil(len(train_dataset)/args.batch_size)
    for epoch in range(args.epochs):
        with tqdm(range(len(train_dataloader))) as pbar:
            for i, imgs in zip(pbar,train_dataloader):
                imgs = imgs.to(device=args.device)
                x = vqgan.get_input(imgs, vqgan.image_key)
                xrec, qloss = vqgan(x)


                loss = 0
                if optimizer_idx == 0:
                    # autoencode
                    aeloss, log_dict_ae = vqgan.loss(qloss, x, xrec, optimizer_idx, vqgan.global_step,
                                                    last_layer=vqgan.get_last_layer(), split="train")

                    loss = aeloss

                if optimizer_idx == 1:
                    # discriminator
                    discloss, log_dict_disc = vqgan.loss(qloss, x, xrec, optimizer_idx, vqgan.global_step,
                                                    last_layer=vqgan.get_last_layer(), split="train")
                    loss = discloss

                loss.backward()
                cmm_loss += loss.item()
                
                    
                    
                if i % args.print_freq == 0:
                    log_vars = {'epoch':epoch, 'iter': i, 'optimnizer_idx':optimizer_idx}
                    log_vars.update({'loss':cmm_loss/args.print_freq})
                    cmm_loss = 0
                    
                    print(log_vars)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")

    parser.add_argument("-n","--name", type=str,const=True,default="", nargs="?",help="postfix for logdir")
    parser.add_argument("-r","--resume",type=str,const=True,default="",nargs="?",help="resume from logdir or checkpoint in logdir")
    parser.add_argument("-b","--base",nargs="*", metavar="base_config.yaml",help="paths to base configs. Loaded from left-to-right. ""Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    parser.add_argument("-t","--train",type=str2bool,const=True,default=False,nargs="?",help="train")
    parser.add_argument("--no-test",type=str2bool,const=True,default=False,nargs="?",help="disable test")
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument("-d","--debug",type=str2bool,nargs="?",const=True,default=False,help="enable post-mortem debugging")
    parser.add_argument("-s","--seed",type=int,default=23,help="seed for seed_everything")
    parser.add_argument("-f","--postfix",type=str,default="",help="post-postfix for default name")

    args = parser.parse_args()
    configs = [OmegaConf.load(cfg) for cfg in args.base]

    vqgan = instantiate_from_config(configs.model)
    train_vqgan = train(configs)


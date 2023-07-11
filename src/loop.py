"""Training loop."""
import torch
import torch.nn as nn
import torchvision.utils as vutils

from utilities.parse import opts
from utilities.torchutils import *
from utilities.utils import *


class TrainLoop():
    """Class to handle training loop."""

    def __init__(
            self, 
            dataset,
            generator: nn.Module, 
            discriminator: nn.Module, 
            writer: SummaryWriter,
            dest_folder: str
            ) -> None:
        """Initialize TrainLoop class.

        @param generator (nn.Module):
            Generator model.
        @param discriminator (nn.Module):
            Discriminator model.
        @param writer (SummaryWriter):
            Tensorboard writer.

        @return (None):
            None.
        """
        super(TrainLoop, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        
        self.dataset = dataset
        
        self.opts = opts
        self.writer = writer
        
        self.optimizer_G = torch.optim.Adam(
                self.generator.parameters(),
                lr=self.opts.lrG,
                betas=(self.opts.beta1, self.opts.beta2)
                )
        self.optimizer_D = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.opts.lrD,
                betas=(self.opts.beta1, self.opts.beta2)
                )

        # exponential weight decay on lr
        self.decayD = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_D, gamma=1-opts.decayD, verbose=True)
        self.decayG = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_G, gamma=1-opts.decayG, verbose=True)

        self.cuda = True if torch.cuda.is_available() else False
        # to avoid unbounding
        self.current_batch_size = 0

        self.dest_folder = dest_folder
       
        if self.cuda:
            self.noise_fixed = torch.randn(size=(self.opts.batch_size, self.opts.latent_dim, 1, 1)).cuda()
            self.noise_vec = lambda size, requires_grad=False: torch.randn(size=size, requires_grad=requires_grad).cuda()
            
            self.x = torch.FloatTensor(opts.batch_size, opts.channels, opts.img_size, opts.img_size).cuda()
            self.x_fake = torch.FloatTensor(opts.batch_size, opts.channels, opts.img_size, opts.img_size).cuda()
            self.y = torch.FloatTensor(opts.batch_size).cuda()
            self.y_fake = torch.FloatTensor(opts.batch_size).cuda()


    def train(self):
        """Train the model."""
        random_sample = generate_random_sample(dataset=self.dataset, batch_size=self.opts.batch_size)

        self.generator.train()
        self.discriminator.train()
            
        for iter in range(0, self.opts.iters):
            
            # NOTE: new
            self.optimizer_G.zero_grad()
            self.optimizer_D.zero_grad()

            if (iter + 1) % self.opts.sample_interval == 0 or iter == 0 or iter + 1 == self.opts.iters:
                fake_test = self.generator(self.noise_fixed)
                # takes arguments of make grid
                if iter == 0:
                    name = f"fake_{iter}.png"
                else:
                    name = f"fake_{iter+1}.png"
                vutils.save_image(
                        tensor=fake_test.detach(),
                        fp=f"{self.dest_folder}/{name}",
                        # normalize=True,  # normalize to range (0, 1)
                        value_range=(0, 255) # scale to range (min, max)
                        )

            # Train Discriminator
            for param in self.discriminator.parameters():
                param.requires_grad = True

            for _ in range(self.opts.itersD):

                self.discriminator.zero_grad()

                batch = random_sample.__next__()
                
                self.current_batch_size = batch.size(dim=0)
                
                self.x.reshape(self.current_batch_size, opts.channels, opts.img_size, opts.img_size)
                
                if self.x.device != 'cpu':
                    self.x = batch.detach().clone().cuda()
                else:
                    self.x = batch.detach().clone()
                
                del batch
                
                if self.opts.noisy_discr_input:
                    self.x += self.noise_vec(size=self.x.shape)

                y_pred = self.discriminator(imgs=self.x).view(-1)
                
                # if instance noise doesn't perform well. uncomment following line to perform label switching
                self.y = self.y.reshape(self.current_batch_size).fill_(1).view(-1)
                self.y_fake = self.y_fake.reshape(self.current_batch_size).fill_(0).view(-1)
                z = self.noise_vec(size=(self.current_batch_size, self.opts.latent_dim, 1, 1))

                imgs_fake = self.generator(z=z)
                # instance noise
                if self.opts.instance_noise:
                    self.y = instance_noise(labels=self.y).view(-1)
                    self.y_fake = instance_noise(labels=self.y_fake).view(-1)

                # label smoothing
                if self.opts.label_smoothing:
                    self.y = label_smoothing(labels=self.y).view(-1)
                    self.y_fake = label_smoothing(self.y_fake).view(-1)

                imgs_fake = self.generator(z=z)

                self.x_fake = self.x_fake.reshape(self.current_batch_size, opts.channels, opts.img_size, opts.img_size)
                self.x_fake = imgs_fake.detach().clone()
                if imgs_fake.device != 'cpu':
                    self.x_fake = imgs_fake.detach().clone().cuda()

                y_pred_fake = self.discriminator(imgs=self.x_fake.detach()).view(-1)
                
                if self.opts.model in ['RSGAN']:
                      d_loss = nn.BCEWithLogitsLoss(reduction='mean')(input=y_pred - y_pred_fake, target=self.y)
                elif self.opts.model in ['RaSGAN']:
                      d_loss = (nn.BCEWithLogitsLoss(reduction='mean')(input=y_pred - torch.mean(y_pred_fake), target=self.y) + nn.BCEWithLogitsLoss(reduction='mean')(input=y_pred_fake - torch.mean(y_pred), target=self.y_fake))/2
                elif self.opts.model in ['RaLSGAN']:
                      d_loss = (torch.mean((y_pred - torch.mean(y_pred_fake) - self.y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) + self.y) ** 2))/2
                else:
                      d_loss = (torch.mean(nn.ReLU()(1.0 - (y_pred - torch.mean(y_pred_fake)))) + torch.mean(nn.ReLU()(1.0 + (y_pred_fake - torch.mean(y_pred)))))/2

                d_loss.backward()
                self.optimizer_D.step()
                # logs discrimnator loss to tensorboard for itersD iteration in one epoch
                self.writer.add_scalar('Loss/Discriminator', d_loss.item(), iter)

            # Train Generator
            for params in self.discriminator.parameters():
                params.requires_grad = False
            for params in self.generator.parameters():
                params.requires_grad = True

            for _ in range(self.opts.itersG):
                self.generator.zero_grad()

                self.y = self.y.reshape(self.current_batch_size).fill_(1).view(-1)
                z = self.noise_vec(size=(self.opts.batch_size, self.opts.latent_dim, 1, 1))

                imgs_fake = self.generator(z=z)
                y_pred_fake = self.discriminator(imgs=imgs_fake).view(-1)

                batch = random_sample.__next__()

                self.current_batch_size = batch.size(dim=0)
                self.x.reshape(self.current_batch_size, opts.channels, opts.img_size, opts.img_size)

                if self.x.device != 'cpu':
                    self.x = batch.detach().clone().cuda()
                else:
                    self.x = batch.detach().clone()

                del batch

                # here I don't add noise to discriminator input beacuse I'm training the generator
                y_pred = self.discriminator(imgs=self.x).view(-1)

                if self.opts.model in ['RSGAN']:
                      g_loss = nn.BCEWithLogitsLoss(reduction='mean')(input=y_pred_fake - y_pred, target=self.y)
                elif self.opts.model in ['RaSGAN']:
                      self.y_fake.reshape(self.current_batch_size).fill_(0)
                      g_loss = (nn.BCEWithLogitsLoss(reduction='mean')(input=y_pred - torch.mean(y_pred_fake), target=self.y_fake) + nn.BCEWithLogitsLoss(reduction='mean')(input=y_pred_fake - torch.mean(y_pred), target=self.y))/2
                elif self.opts.model in ['RaLSGAN']:
                      g_loss = (torch.mean((y_pred - torch.mean(y_pred_fake) + self.y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) - self.y) ** 2))/2
                else:
                      g_loss = (torch.mean(nn.ReLU()(1.0 + (y_pred - torch.mean(y_pred_fake)))) + torch.mean(nn.ReLU()(1.0 - (y_pred_fake - torch.mean(y_pred)))))/2

                g_loss.backward()
                self.optimizer_G.step()
                # logs generator loss to tensorboard for itersG iteration in one epoch
                self.writer.add_scalar('Loss/Generator', g_loss.item(), iter)

            self.decayD.step()
            self.decayG.step()

            print(
                    "[Epoch %d/%d] D_loss: %.4f, G_loss: %.4f"
                    % (iter + 1, self.opts.iters, d_loss.item(), g_loss.item())
                )

        self.writer.close()

"""# GAN MODEL"""

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    nn.init.xavier_uniform_(m.weight.data).cuda()
    nn.init.constant_(m.bias.data, 0).cuda()

learning_rate_G = 0.03
learning_rate_D = 0.03

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.main = nn.Sequential(
            nn.Linear(100, 100),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input).cuda()

netG = Generator()
if use_cuda: netG = netG.cuda()
netG.apply(weights_init)

class Discriminator(nn.Module):
    def __init__(self):
      super(Discriminator, self).__init__()
      self.use_cuda = torch.cuda.is_available()
      self.main = nn.Sequential(
        nn.Linear(100, 2),
        nn.Softmax(dim = 1)
      ) 
      
    def forward(self, input):
      return self.main(input).cuda()

netD = Discriminator()
if use_cuda: netD = netD.cuda()
netD.apply(weights_init)

real_label = torch.tensor([0,1])
fake_label = torch.tensor([1,0])
optimizerD = optim.Adadelta(netD.parameters(), lr=learning_rate_G)
optimizerG = optim.Adadelta(netG.parameters(), lr=learning_rate_D)

class Dropout_layer(nn.Module):
  def __init__(self):
    super(Dropout_layer, self).__init__()
    self.d = nn.Dropout(p=0.5)
    
  def forward(self, input):
    return self.d(input).cuda()

dropout_layer = Dropout_layer()
if use_cuda: dropout_layer = dropout_layer.cuda()

class Final_layer(nn.Module):
    def __init__(self):
        super(Final_layer, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.main = nn.Sequential(
            nn.Linear(101, 2),
            nn.Softmax(dim = 1)
        )
    def forward(self, input):
        return self.main(input).cuda()

net_final = Final_layer()
if use_cuda: net_final = net_final.cuda()
net_final.apply(weights_init)

final_par = list(net_final.parameters())

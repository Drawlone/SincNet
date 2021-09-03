# speaker_id.py
# Mirco Ravanelli 
# Mila - University of Montreal 

# July 2018

# Description: 
# This code performs a speaker_id experiments with SincNet.

# How to run it:
# python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg

import os

import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from data_io import ReadList, read_conf_inp
from dnn_models import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
# Reading cfg file
options = read_conf_inp("cfg/SincNet_CNCeleb.cfg")


class AudioDataset(Dataset):
    def __init__(self, annotations_file, id_num, data_dir, croplen):
        """
        Parameters
        ----------
        annotations_file 数据集列表
        data_dir 数据集根目录
        """
        super(AudioDataset, self).__init__()
        self.data_dir = data_dir
        self.croplen = croplen
        self.id_num = id_num
        with open(annotations_file, 'r') as f:
            self.datalist = f.readlines()

    def __getitem__(self, index) -> T_co:
        item = self.datalist[index].replace('\n', "")
        audio_path = os.path.join(self.data_dir, item)
        audio, _ = sf.read(audio_path)
        label = item.split("/")[0][2:7]
        channels = len(audio.shape)
        if channels != 1:
            print('WARNING: stereo to mono: ' + audio_path)
            audio = audio[:, 0]

        # accessing to a random chunk
        audio_len = audio.shape[0]
        audio_beg = np.random.randint(audio_len - self.croplen - 1)
        audio_end = audio_beg + self.croplen

        audio_chunk = audio[audio_beg:audio_end]
        # y_one_hot = torch.zeros([self.id_num])
        # y_one_hot.scatter_(0, torch.LongTensor([int(label)]), 1)
        return torch.tensor(audio_chunk, dtype=torch.float32), torch.tensor(int(label))

    def __len__(self):
        return len(self.datalist)


# [data]
tr_lst = options.tr_lst
te_lst = options.te_lst
pt_file = options.pt_file
class_dict_file = options.lab_dict
data_folder = options.data_folder + '/'
output_folder = options.output_folder

# [optimization]
lr = float(options.lr)
batch_size = int(options.batch_size)
N_epochs = int(options.N_epochs)
N_batches = int(options.N_batches)
N_eval_epoch = int(options.N_eval_epoch)
seed = int(options.seed)

# training list
wav_lst_tr = ReadList(tr_lst)
snt_tr = len(wav_lst_tr)

# test list
wav_lst_te = ReadList(te_lst)
snt_te = len(wav_lst_te)

# setting seed
torch.manual_seed(seed)
np.random.seed(seed)


# loss function


# Converting context and shift in samples
# wshift = int(fs * cw_shift / 1000.00)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Batch_dev
batch_size = 16

# Loading label dictionary
# lab_dict = np.load(class_dict_file, allow_pickle=True).item()

model = NeuralNetwork(options).to(device)
current_epoch = 0
if os.path.exists(output_folder + "checkpoint.cpt"):
    cpt = torch.load(output_folder + "checkpoint.cpt")
    current_epoch = cpt["epoch"]
    model.load_state_dict(torch.load(output_folder + f"/model_{current_epoch}.pth"))
    print("[*] load model from " + output_folder + f"/model_{current_epoch}.pth")
loss_fn = nn.NLLLoss()
optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.95, eps=1e-8)

training_data = AudioDataset("train.lst", 7, "data", model.audio_len)
train_dataloader = DataLoader(training_data, batch_size=batch_size)
for X, y in train_dataloader:
    print("Shape of X [N, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

epochs = 3
for t in range(current_epoch + 1, epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    checkpoint = {
        'epoch': t
    }
    torch.save(checkpoint, output_folder + "/checkpoint.cpt")
    torch.save(model.state_dict(), output_folder + f"/model_{t}.pth")
    print(f"Saved PyTorch Model State to model_{t}.pth")
print("Done!")

'''
for epoch in range(N_epochs):

    test_flag = 0
    CNN_net.train()
    DNN1_net.train()
    DNN2_net.train()

    loss_sum = 0
    err_sum = 0

    for i in range(N_batches):
        
        # [inp, lab] = create_batches_rnd(batch_size, data_folder, wav_lst_tr, snt_tr, wlen, lab_dict, 0.2)
        pout = DNN2_net(DNN1_net(CNN_net(inp)))

        pred = torch.max(pout, dim=1)[1]
        loss = cost(pout, lab.long())
        err = torch.mean((pred != lab.long()).float())

        optimizer_CNN.zero_grad()
        optimizer_DNN1.zero_grad()
        optimizer_DNN2.zero_grad()

        loss.backward()
        optimizer_CNN.step()
        optimizer_DNN1.step()
        optimizer_DNN2.step()

        loss_sum = loss_sum + loss.detach()
        err_sum = err_sum + err.detach()

    loss_tot = loss_sum / N_batches
    err_tot = err_sum / N_batches

    # Full Validation  new
    if epoch % N_eval_epoch == 0:

        CNN_net.eval()
        DNN1_net.eval()
        DNN2_net.eval()
        test_flag = 1
        loss_sum = 0
        err_sum = 0
        err_sum_snt = 0

        with torch.no_grad():
            for i in range(snt_te):
                [signal, fs] = sf.read(data_folder + wav_lst_te[i])

                signal = torch.from_numpy(signal).float().cuda().contiguous()
                lab_batch = lab_dict[wav_lst_te[i]]

                # split signals into chunks
                beg_samp = 0
                end_samp = wlen

                N_fr = int((signal.shape[0] - wlen) / (wshift))

                sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()
                lab = Variable((torch.zeros(N_fr + 1) + lab_batch).cuda().contiguous().long())
                pout = Variable(torch.zeros(N_fr + 1, class_lay[-1]).float().cuda().contiguous())
                count_fr = 0
                count_fr_tot = 0
                while end_samp < signal.shape[0]:
                    sig_arr[count_fr, :] = signal[beg_samp:end_samp]
                    beg_samp = beg_samp + wshift
                    end_samp = beg_samp + wlen
                    count_fr = count_fr + 1
                    count_fr_tot = count_fr_tot + 1
                    if count_fr == Batch_dev:
                        inp = Variable(sig_arr)
                        pout[count_fr_tot - Batch_dev:count_fr_tot, :] = DNN2_net(DNN1_net(CNN_net(inp)))
                        count_fr = 0
                        sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()

                if count_fr > 0:
                    inp = Variable(sig_arr[0:count_fr])
                    pout[count_fr_tot - count_fr:count_fr_tot, :] = DNN2_net(DNN1_net(CNN_net(inp)))

                pred = torch.max(pout, dim=1)[1]
                loss = cost(pout, lab.long())
                err = torch.mean((pred != lab.long()).float())

                [val, best_class] = torch.max(torch.sum(pout, dim=0), 0)
                err_sum_snt = err_sum_snt + (best_class != lab[0]).float()

                loss_sum = loss_sum + loss.detach()
                err_sum = err_sum + err.detach()

            err_tot_dev_snt = err_sum_snt / snt_te
            loss_tot_dev = loss_sum / snt_te
            err_tot_dev = err_sum / snt_te

        print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (
            epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

        with open(output_folder + "/res.res", "a") as res_file:
            res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (
                epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

        checkpoint = {'CNN_model_par': CNN_net.state_dict(),
                      'DNN1_model_par': DNN1_net.state_dict(),
                      'DNN2_model_par': DNN2_net.state_dict(),
                      }
        torch.save(checkpoint, output_folder + '/model_raw.pkl')

    else:
        print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot, err_tot))
'''

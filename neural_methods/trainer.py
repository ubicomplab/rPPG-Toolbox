import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


def gen_plot(bvp, label):
    fig = plt.figure(1, figsize=(8, 6))
    plt.plot(bvp[0], 'b', label="predict")
    plt.plot(label[0], 'r', label="label")
    plt.legend()
    return fig

# Physnet


def train_physnet(model, loss_model, batch, device, train_step, twriter):
    rPPG, x_visual, x_visual3232, x_visual1616 = model(
        Variable(batch[0]).to(torch.float32).to(device))
    BVP_label = Variable(batch[1]).to(torch.float32).to(device)
    rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
    BVP_label = (BVP_label - torch.mean(BVP_label)) / \
        torch.std(BVP_label)  # normalize
    loss_ecg = loss_model(rPPG, BVP_label)
    loss_ecg.backward()
    twriter.add_scalar("train_loss", scalar_value=float(
        loss_ecg), global_step=train_step)
    return loss_ecg


def valid_physnet(model, loss_model, batch, device, valid_step, twriter):
    BVP_label = Variable(batch[1]).to(torch.float32).to(device)
    rPPG, x_visual, x_visual3232, x_visual1616 = model(
        Variable(batch[0]).to(torch.float32).to(device))
    rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
    BVP_label = (BVP_label - torch.mean(BVP_label)) / \
        torch.std(BVP_label)  # normalize
    loss_ecg = loss_model(rPPG, BVP_label)
    twriter.add_scalar("valid_loss", scalar_value=float(
        loss_ecg), global_step=valid_step)
    return loss_ecg


def test_physnet(model, loss_model, batch, device, test_step, twriter):
    BVP_label = Variable(batch[1]).to(torch.float32).to(device)
    rPPG, x_visual, x_visual3232, x_visual1616 = model(
        Variable(batch[0]).to(torch.float32).to(device))
    rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
    BVP_label = (BVP_label - torch.mean(BVP_label)) / \
        torch.std(BVP_label)  # normalize
    loss_ecg = loss_model(rPPG, BVP_label)
    twriter.add_scalar("test_loss", scalar_value=float(
        loss_ecg), global_step=test_step)
    figure = gen_plot(rPPG.cpu().numpy(), BVP_label.cpu().numpy())
    twriter.add_figure("test_img", figure, global_step=test_step)
    return loss_ecg

# rPPGnet


def train_rppgnet(model, loss_model, batch, device, train_step, twriter):
    BVP_label = Variable(batch[1]).to(torch.float32).to(device)
    skin_map, rPPG_aux, rPPG, rPPG_SA1, rPPG_SA2, rPPG_SA3, rPPG_SA4, x_visual6464, x_visual3232 = model(
        Variable(batch[0]).to(torch.float32).to(device))
    # loss_binary = criterion_Binary(skin_map, skin_seg_label)

    rPPG = (rPPG-torch.mean(rPPG)) / torch.std(rPPG)	 	# normalize2
    rPPG_SA1 = (rPPG_SA1-torch.mean(rPPG_SA1)) / \
        torch.std(rPPG_SA1)	 	# normalize2
    rPPG_SA2 = (rPPG_SA2-torch.mean(rPPG_SA2)) / \
        torch.std(rPPG_SA2)	 	# normalize2
    rPPG_SA3 = (rPPG_SA3-torch.mean(rPPG_SA3)) / \
        torch.std(rPPG_SA3)	 	# normalize2
    rPPG_SA4 = (rPPG_SA4-torch.mean(rPPG_SA4)) / \
        torch.std(rPPG_SA4)	 	# normalize2
    rPPG_aux = (rPPG_aux-torch.mean(rPPG_aux)) / \
        torch.std(rPPG_aux)	 	# normalize2

    loss_ecg = loss_model(rPPG, BVP_label)
    loss_ecg1 = loss_model(rPPG_SA1, BVP_label)
    loss_ecg2 = loss_model(rPPG_SA2, BVP_label)
    loss_ecg3 = loss_model(rPPG_SA3, BVP_label)
    loss_ecg4 = loss_model(rPPG_SA4, BVP_label)
    loss_ecg_aux = loss_model(rPPG_aux, BVP_label)

    loss = 0.1*0 + 0.5 * \
        (loss_ecg1 + loss_ecg2 + loss_ecg3 + loss_ecg4 + loss_ecg_aux) + loss_ecg
    loss.backward()
    twriter.add_scalar("train_loss", scalar_value=float(
        loss), global_step=train_step)
    return loss


def valid_rppgnet(model, loss_model, batch, device, valid_step, twriter):
    BVP_label = Variable(batch[1]).to(torch.float32).to(device)
    skin_map, rPPG_aux, rPPG, rPPG_SA1, rPPG_SA2, rPPG_SA3, rPPG_SA4, x_visual6464, x_visual3232 = model(
        Variable(batch[0]).to(torch.float32).to(device))
    # loss_binary = criterion_Binary(skin_map, skin_seg_label)

    rPPG = (rPPG-torch.mean(rPPG)) / torch.std(rPPG)	 	# normalize2
    rPPG_SA1 = (rPPG_SA1-torch.mean(rPPG_SA1)) / \
        torch.std(rPPG_SA1)	 	# normalize2
    rPPG_SA2 = (rPPG_SA2-torch.mean(rPPG_SA2)) / \
        torch.std(rPPG_SA2)	 	# normalize2
    rPPG_SA3 = (rPPG_SA3-torch.mean(rPPG_SA3)) / \
        torch.std(rPPG_SA3)	 	# normalize2
    rPPG_SA4 = (rPPG_SA4-torch.mean(rPPG_SA4)) / \
        torch.std(rPPG_SA4)	 	# normalize2
    rPPG_aux = (rPPG_aux-torch.mean(rPPG_aux)) / \
        torch.std(rPPG_aux)	 	# normalize2

    loss_ecg = loss_model(rPPG, BVP_label)
    loss_ecg1 = loss_model(rPPG_SA1, BVP_label)
    loss_ecg2 = loss_model(rPPG_SA2, BVP_label)
    loss_ecg3 = loss_model(rPPG_SA3, BVP_label)
    loss_ecg4 = loss_model(rPPG_SA4, BVP_label)
    loss_ecg_aux = loss_model(rPPG_aux, BVP_label)

    # loss = 0.1*loss_binary + 0.5 * \
    #     (loss_ecg1 + loss_ecg2 + loss_ecg3 + loss_ecg4 + loss_ecg_aux) + loss_ecg
    loss = 0.1*0 + 0.5 * \
        (loss_ecg1 + loss_ecg2 + loss_ecg3 + loss_ecg4 + loss_ecg_aux) + loss_ecg
    twriter.add_scalar("valid_loss", scalar_value=float(
        loss), global_step=valid_step)
    return loss


def test_rppgnet(model, loss_model, batch, device, test_step, twriter):
    BVP_label = Variable(batch[1]).to(torch.float32).to(device)
    skin_map, rPPG_aux, rPPG, rPPG_SA1, rPPG_SA2, rPPG_SA3, rPPG_SA4, x_visual6464, x_visual3232 = model(
        Variable(batch[0]).to(torch.float32).to(device))
    # loss_binary = criterion_Binary(skin_map, skin_seg_label)

    rPPG = (rPPG-torch.mean(rPPG)) / torch.std(rPPG)	 	# normalize2
    rPPG_SA1 = (rPPG_SA1-torch.mean(rPPG_SA1)) / \
        torch.std(rPPG_SA1)	 	# normalize2
    rPPG_SA2 = (rPPG_SA2-torch.mean(rPPG_SA2)) / \
        torch.std(rPPG_SA2)	 	# normalize2
    rPPG_SA3 = (rPPG_SA3-torch.mean(rPPG_SA3)) / \
        torch.std(rPPG_SA3)	 	# normalize2
    rPPG_SA4 = (rPPG_SA4-torch.mean(rPPG_SA4)) / \
        torch.std(rPPG_SA4)	 	# normalize2
    rPPG_aux = (rPPG_aux-torch.mean(rPPG_aux)) / \
        torch.std(rPPG_aux)	 	# normalize2

    loss_ecg = loss_model(rPPG, BVP_label)
    loss_ecg1 = loss_model(rPPG_SA1, BVP_label)
    loss_ecg2 = loss_model(rPPG_SA2, BVP_label)
    loss_ecg3 = loss_model(rPPG_SA3, BVP_label)
    loss_ecg4 = loss_model(rPPG_SA4, BVP_label)
    loss_ecg_aux = loss_model(rPPG_aux, BVP_label)

    # loss = 0.1*loss_binary + 0.5 * \
    #     (loss_ecg1 + loss_ecg2 + loss_ecg3 + loss_ecg4 + loss_ecg_aux) + loss_ecg
    loss = 0.1*0 + 0.5 * \
        (loss_ecg1 + loss_ecg2 + loss_ecg3 + loss_ecg4 + loss_ecg_aux) + loss_ecg
    twriter.add_scalar("test_loss", scalar_value=float(
        loss), global_step=test_step)
    figure = gen_plot(rPPG.cpu().numpy(), BVP_label.cpu().numpy())
    twriter.add_figure("test_img", figure, global_step=test_step)
    return loss

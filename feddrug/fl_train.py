import os
import time
from multiprocessing import Process
from typing import Tuple
from collections import OrderedDict
import fitlog
import flwr as fl
from flwr.server import strategy
import numpy as np
from flwr.server.strategy import FedAvg, FedAdagrad
import torch
from feddrug.data.gen_iid_data import load_fed_data
from core.train_utils import init_featurizer, load_model, mkdir_p
from feddrug.utils import run_fedavg_epoch, run_eval_epoch
import feddrug.app
import flwr.common
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASET = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def start_server(args, return_dict):
    """Start the server with a slightly adjusted FedAvg strategy."""
    import feddrug
    strategy_name = args['strategy']
    num_rounds = args['num_rounds']
    num_clients = args['num_clients']
    fraction_fit = args['fraction_fit']
    eta_l = args['lr']
    eta = args['eta']
    if strategy_name == "fedavg":
        strategy = FedAvg(min_available_clients=num_clients, fraction_fit=fraction_fit)
    elif strategy_name == 'fedada':
        model = load_model(args).to('cpu')
        x = [ 1e-9*torch.zeros_like(val).cpu().numpy() if val.nelement() > 1 else val.view(-1).cpu().numpy() for _, val in model.state_dict().items()]
        strategy = FedAdagrad(min_available_clients=num_clients, fraction_fit=fraction_fit, initial_parameters=flwr.common.weights_to_parameters(x),
                    eta_l=eta_l, eta=eta)
    # Exposes the server by default on port 8080
    hist = feddrug.app.start_server(strategy=strategy, config={"num_rounds": num_rounds})
    return_dict['hist'] = hist

def start_client(args, dataset) -> None:
    """Start a single client with the provided dataset."""

    # Load and compile a Keras model for CIFAR-10
    #net = SimpleNet().to(DEVICE)
    model = load_model(args).to(DEVICE)
    # Unpack the CIFAR-10 dataset partition
    trainloader, testloader = dataset

    class DrugClient(fl.client.NumPyClient):
        def get_parameters(self):
            x = [ val.cpu().numpy() if val.nelement() > 1 else val.view(-1).cpu().numpy() for _, val in model.state_dict().items()]
            return x

        def set_parameters(self, parameters):
            USE_FEDBN = False
            if USE_FEDBN:
                keys = [k for k in model.state_dict().keys() if "batches" not in k]
                params_dict = zip(keys, parameters)
                state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=False)
            else:
                params_dict = zip(model.state_dict().keys(), parameters)
                state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)
            model.train()
        def fit(self, parameters, config):
            self.set_parameters(parameters)
            run_fedavg_epoch(args, model, trainloader)
            #train(net, trainloader, epochs=1)
            return self.get_parameters(), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            rmse = run_eval_epoch(args, model, testloader)
            return float(rmse), len(testloader), {"accuracy": float(rmse)}

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=DrugClient())


def run_simulation(args):
    """Start a FL simulation."""
    # This will hold all the processes which we are going to create
    processes = []
    mannager = torch.multiprocessing.Manager()
    return_dict = mannager.dict()
    # Start the server
    server_process = Process(
        target=start_server, args=(args, return_dict)
    )
    server_process.start()
    processes.append(server_process)

    # Optionally block the script here for a second or two so the server has time to start
    time.sleep(2)

    # Load the dataset partitions
    #partitions = dataset.load(num_clients)
    partitions, test_dl = load_fed_data(args)
    # Start all the clients
    for partition in partitions:
        client_process = Process(target=start_client, args=(args, partition,))
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()
    fitlog.set_log_dir('logs')
    fitlog.add_hyper(args)
    hist = return_dict['hist']
    for step, loss in hist.losses_distributed:
        fitlog.add_loss(loss,name="Loss",step=step)
        fitlog.add_metric({"dev":{"rmse":loss}}, step=step)
    fitlog.finish()

if __name__ == "__main__":
    import sys
    args_file = sys.argv[1]
    torch.multiprocessing.set_start_method("spawn")
    from core.utils import init_args
    args = init_args(args_file)
    run_simulation(args)
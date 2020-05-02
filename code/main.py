import argparse
from copy import deepcopy
import torch
import temp_config_files
import optimizers
from neural_networks_architecture import MLP, CNN, fit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_epochs', type=int, default=30)
    parser.add_argument('-dataset', type=str, default='cifar')
    parser.add_argument('-num_train', type=int, default=50000)
    parser.add_argument('-num_val', type=int, default=2048)
    parser.add_argument('-lr_schedule', type=bool, default=True)
    parser.add_argument('-only_plot', type=bool, default=True)
    args = parser.parse_args()

    data = getattr(temp_config_files, 'load_'+args.dataset)(
        num_train=args.num_train,
        num_val=args.num_val
    )
    print(f'Loaded data partitions: ({len(data[0])}), ({len(data[1])})')
    opt_tasks = [
        'padam','amsgrad',
        'adamax','adabound','adabound_w',
        'sgd','sgd_momentum','sgd_nesterov',
        'sgd_weight_decay','sgd_lrd','rmsprop',
        'adam','adam_l2','adamW','adam_lrd',
        'Radam','RadamW','Radam_lrd',
        'nadam','lookahead_sgd','lookahead_adam',
        'gradnoise_adam','graddropout_adam','hessian_free'
    ]
    opt_losses, opt_val_losses, opt_labels = [], [], []
    def do_stuff(opt):
        print(f'\nTraining {opt} for {args.num_epochs} epochs...')
        net = CNN() if args.dataset == 'cifar' else MLP()
        _, kwargs = temp_config_files.split_optim_dict(temp_config_files.optim_dict[opt])
        optimizer = temp_config_files.task_to_optimizer(opt)(
            params=net.parameters(),
            **kwargs
        )
        optimizer = temp_config_files.wrap_optimizer(opt, optimizer)
        return fit(net, data, optimizer, num_epochs=args.num_epochs, lr_schedule=True,hessian_free=False)

    for opt in opt_tasks:
        if args.only_plot:
            losses = temp_config_files.load_losses(dataset=args.dataset, filename=opt)
            val_losses = temp_config_files.load_losses(dataset=args.dataset, filename=opt+'_val')
            losses, val_losses = do_stuff(opt)
            temp_config_files.save_losses(losses, dataset=args.dataset, filename=opt)
            temp_config_files.save_losses(val_losses, dataset=args.dataset, filename=opt+'_val')
        if losses is not None:
            opt_losses.append(losses)
            opt_val_losses.append(val_losses)
            opt_labels.append(temp_config_files.split_optim_dict(temp_config_files.optim_dict[opt])[0])

    if not torch.cuda.is_available():
        assert len(opt_losses) == len(opt_val_losses)
        temp_config_files.plot_losses(
            losses=opt_losses,
            val_losses=opt_val_losses,
            labels=opt_labels,
            num_epochs=args.num_epochs,
            title=args.dataset,
            plot_val=False,
            yscale_log=False,
            max_epochs=30
        )

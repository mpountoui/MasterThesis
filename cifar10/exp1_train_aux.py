from nn.nn_utils                import load_model, save_model
from loaders.cifar_dataset      import cifar10_loader
from models.cifar_tiny          import Cifar_Tiny
from models.resnet              import ResNet18
from nn.retrieval_evaluation    import evaluate_model_retrieval
from nn.pkt_transfer            import prob_transfer
from Transformers.MyTransformer import ViT

import torch

try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

Path = '/Users/ioannisbountouris/PycharmProjects/MasterThesis'

if IN_COLAB :
    import sys
    sys.path.append('/content/MasterThesis')
    Path = '/content/MasterThesis'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_transfer(learning_rates=(0.001, 0.0001), iters=(3, 0), method='mds'):
    torch.manual_seed(12345)
    student_layers, teacher_layers, weights, loss_params, T = (2,), (3,), (1,), {}, 2
    print(method)
    transfer_name = method

    # Output paths
    output_path  = Path + '/cifar10/models/aux_'  + transfer_name + '.model'
    results_path = Path + '/cifar10/results/aux_' + transfer_name

    student_net = ViT((3, 32, 32), n_patches=8, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)

    # Load the teacher model
    teacher_net = ResNet18(num_classes=10)
    load_model(teacher_net, Path + '/cifar10/models/resnet18_cifar10.model')

    train_loader, test_loader, train_loader_raw = cifar10_loader(batch_size=128)

    student_net.to(device)
    teacher_net.to(device)

    # Perform the transfer
    W = None
    for lr, iters in zip(learning_rates, iters):

        if method == 'pkt':
            kernel_parameters = {'student': 'combined', 'teacher': 'combined', 'loss': 'combined'}
            prob_transfer(student_net, teacher_net, train_loader, epochs=iters, lr=lr,
                          teacher_layers=teacher_layers, student_layers=student_layers, layer_weights=weights,
                          kernel_parameters=kernel_parameters, loss_params=loss_params)
        else:
            assert False

    save_model(student_net, output_path)
    print("Model saved at ", output_path)

    # Perform the evaluation
    evaluate_model_retrieval(net=Cifar_Tiny(num_classes=10), path=output_path,
                             result_path=results_path + '_retrieval.pickle', layer=3)
    evaluate_model_retrieval(net=Cifar_Tiny(num_classes=10), path=output_path,
                             result_path=results_path + '_retrieval_e.pickle', layer=3, metric='l2')


if __name__ == '__main__':
    run_transfer(iters=(30, 10), method='pkt')


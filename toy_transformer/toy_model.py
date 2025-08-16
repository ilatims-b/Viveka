import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Literal
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.train import HookedTransformerTrainConfig, train
from mealymarkov import MarkovMealyModel

class MarkovData(Dataset):
    def __init__(
        self,
        n_gen: int,
        gen_len: int,
        n_states: int,
        d_vocab: int,
        T_list: list[np.ndarray],
        eta0: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        seed: int = 42
    ):
        self.model = MarkovMealyModel(n_states, d_vocab, T_list, eta0, rng)
        self.gen_len = gen_len
        self.data = []
        self.states = []
        for i in range(n_gen):
            tokens, states = self.model.sample_sequence(max_new_tokens=gen_len, seed=seed)
            self.data.append(torch.tensor(tokens, dtype=torch.int64))
            self.states.append(states)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'tokens': self.data[idx]}


def train_model(
    # Dataset
    dataset: MarkovData,

    # Transformer Architecture
    n_layers: int = 4,
    d_model: int = 64,
    n_heads: int = 1,
    d_head: int = 8,
    attn_only: bool = False,
    d_mlp: int = 256,
    act_fn: Literal['relu', 'gelu', 'silu', 'gelu_new', 'solu_ln', 'gelu_fast'] = 'relu',
    normalization_type: Literal['LN', 'LNPre', 'RMS', 'RMSPre'] = 'LN',
    positional_embedding_type: Literal['standard', 'rotary', 'shortformer'] = 'standard',

    # Training Hyperparameters
    n_epochs: int = 1e6,
    batch_size: int = 64,
    lr: float = 1e-2,
    optimizer_name: Literal['Adam', 'AdamW', 'SGD'] = 'SGD',

    # System / IO
    device: str = "cpu",
    seed: int = 42,
    save_every: int = 1000,
    save_dir: str = "./checkpoints",
    print_every: int = 100
) -> HookedTransformer:
    """
    Train a HookedTransformer on sequences generated from a Mealy Markov model.

    This function constructs a HookedTransformer model with the given 
    architecture and optimization hyperparameters, and trains it on 
    sequences generated from a custom Markov process dataset.

    Parameters
    ----------
    dataset : MarkovData
        Training dataset containing token sequences generated from a 
        Mealy-Markov process.

    n_layers : int
        Number of transformer layers.
    d_model : int
        Dimension of the model (embedding and hidden sizes).
    n_heads : int
        Number of attention heads.
    d_head : int
        Dimension per attention head.
    attn_only : bool
        Whether the transformer is attention-only, without any MLP blocks.
    d_mlp : int
        Dimension of the feedforward hidden layer.
    act_fn : {'relu', 'gelu', 'silu', 'gelu_new', 'solu_ln', 'gelu_fast'}
        Activation function used in MLP layers.
    normalization_type : {'LN', 'LNPre', 'RMS', 'RMSPre'}
        Normalization strategy applied in transformer layers.
    positional_embedding_type : {'standard', 'rotary', 'shortformer'}
        Type of positional embeddings used in the model.

    n_epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate for optimization.
    optimizer_name : {'Adam', 'AdamW', 'SGD'}
        Optimizer to use.
    
    device : str
        Device where the model will be trained (e.g., ``"cpu"``, ``"cuda"``).
    seed : int
        Random seed for reproducibility
    save_every : int
        Frequency (in steps) to checkpoint the model.
    save_dir : str
        Directory where checkpoints will be saved.
    print_every : int
        Frequency (in steps) to log training progress.

    Returns
    -------
    HookedTransformer
        The trained transformer model.
    """

    d_vocab = dataset.model.V
    n_ctx = dataset.gen_len
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_ctx=n_ctx,
        d_head=d_head,
        n_heads=n_heads,
        d_mlp=d_mlp,
        act_fn=act_fn,
        d_vocab=d_vocab,
        attn_only=attn_only,
        normalization_type=normalization_type,
        device=device,
        positional_embedding_type=positional_embedding_type,
        seed=seed,
        default_prepend_bos=False,
    )

    model = HookedTransformer(cfg, move_to_device=True)

    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if save_dir is not None:
        torch.save(cfg, save_dir + '/model_cfg.pt')

    train_cfg = HookedTransformerTrainConfig(
        num_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer_name=optimizer_name,
        device=device,
        seed=seed,
        save_every=save_every,
        save_dir=save_dir,
        print_every=print_every
    )

    return train(model, train_cfg, dataset)

def finetune_model(
    model: HookedTransformer,
    dataset: MarkovData,
    n_epochs: int,
    batch_size: int = 64,
    lr: float = 1e-2,
    optimizer_name: Literal['Adam', 'AdamW', 'SGD'] = 'SGD',
    device: str = "cpu",
    seed: int = 42,
    save_every: int = 1000,
    save_dir: str = "./checkpoints",
    print_every: int = 100
) -> HookedTransformer:
    """
    Finetune a pretrained HookedTransformer on sequences generated from a Mealy Markov model.

    Parameters
    ----------
    model : HookedTransformer
        A pre-trained model to finetune.
    dataset : MarkovData
        Training dataset containing token sequences generated from a 
        Mealy-Markov process.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate for optimization.
    optimizer_name : {'Adam', 'AdamW', 'SGD'}
        Optimizer to use.    
    device : str
        Device where the model will be trained (e.g., ``"cpu"``, ``"cuda"``).
    seed : int
        Random seed for reproducibility
    save_every : int
        Frequency (in steps) to checkpoint the model.
    save_dir : str
        Directory where checkpoints will be saved.
    print_every : int
        Frequency (in steps) to log training progress.

    Returns
    -------
    HookedTransformer
        The trained transformer model.
    """
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cfg = HookedTransformerTrainConfig(
        num_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer_name=optimizer_name,
        device=device,
        seed=seed,
        save_every=save_every,
        save_dir=save_dir,
        print_every=print_every
    )

    return train(model, cfg, dataset)

def load_model(model_path, cfg_path) -> HookedTransformer:
    '''Loads a model into HookedTransformer given the path where its weights are stored.'''
    if not (os.path.exists(model_path) and os.path.exists(cfg_path)):
        raise ValueError('Path doesn\'t exist.')
    model = HookedTransformer(torch.load(cfg_path, weights_only=False))
    model.load_state_dict(torch.load(model_path))
    return model

if __name__ == '__main__':
    T0 = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0.5, 0, 0]
    ])
    T1 = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0.5, 0, 0]
    ])
    dataset = MarkovData(n_gen=10000, gen_len=64, n_states=3, d_vocab=2, T_list=[T0, T1])

    if os.path.exists('toy_transformer/checkpoints/model_0.pt'):
        model = load_model('toy_transformer/checkpoints/model_0.pt',
                           'toy_transformer/checkpoints/model_cfg.pt')
    else:
        model = train_model(
            dataset=dataset,
            n_epochs=5,
            save_every=1000,
            print_every=1000,
            save_dir='toy_transformer/checkpoints'
        )

    model_2 = finetune_model(model, dataset, 5, save_dir=None)

    logits = model(torch.tensor([[0,1,1,0,1,0,0,1,1,0],
                                 [1,0,1,1,0,1,0,0,1,1],
                                 [1,0,0,1,0,0,1,0,0,1]], dtype=torch.int64))
    print(logits[:, -1, :])
    print(logits[:, -1, :].argmax(dim=-1))
    # Ground truth values: [1, 0, R]

    print()
    sample, states = dataset.model.sample_sequence(max_new_tokens=40)
    preds = model(torch.tensor(sample, dtype=torch.int64)).argmax(dim=-1).flatten().tolist()
    for s, pred in zip(sample[1:], preds[:-1]):
        print(f'Actual: {s}, Predicted: {pred}')

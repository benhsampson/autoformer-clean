from dataset.ett_small import ETTSmall_Dataset, ETTSmall_DatasetConfig
from models.Autoformer import AutoformerConfig
from trainer import TrainConfig, Trainer


def main():
    SEQ_LEN = 96
    LABEL_LEN = 48
    PRED_LEN = 24
    # NOTE: ETTSmall_Dataset does not support TimeF encoding
    dataset = ETTSmall_Dataset(
        ETTSmall_DatasetConfig(
            seq_len=SEQ_LEN,
            label_len=LABEL_LEN,
            pred_len=PRED_LEN,
            scale=True,
            highest_freq='minute',
            task='M',
            target=None,
            root_path='data/ETDataset/ETT-small',
            data_path='ETTh1.csv',
        )
    )
    train_cfg = TrainConfig(
        model='autoformer',
        model_cfg=AutoformerConfig(
            seq_len=SEQ_LEN,
            label_len=LABEL_LEN,
            pred_len=PRED_LEN,
            output_attention=True,
            moving_avg=25,
            enc_in=dataset.num_features,
            dec_in=dataset.num_features,
            d_model=512,
            embed='fixed',
            freq='h',
            dropout=0.05,
            factor=3,
            n_heads=8,
            d_ff=1024,
            activation='gelu',
            e_layers=2,
            d_layers=1,
            c_out=dataset.num_features,
        ),
        dataset=dataset,
        splits=(3 / 6, 1 / 6, 2 / 6),
        loss_fn='mse',
        optimizer='adam',
        lradj='type1',
        use_gpu=True,
        checkpoints_path='checkpoints',
        train_epochs=8,
        batch_size=32,
        learning_rate=1e-4,
        patience=3,
        delta=0.0,
    )
    trainer = Trainer(train_cfg)
    # trainer.train()
    trainer.test()


if __name__ == '__main__':
    main()

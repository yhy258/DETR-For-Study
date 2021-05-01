class ModelConfig:
    d_model = 512
    nhead = 8
    num_encoder_layers = 6,
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1

class TrainConfig :
    predict_num = 100
    classes_num = 100
    epochs = 100
    learning_rate = 1e-3
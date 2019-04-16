
class Config:
    """dir & path"""
    train_filename = "./data/libffm_toy/criteo.tr.r100.gbdt0.ffm"
    val_filename = "./data/libffm_toy/criteo.va.r100.gbdt0.ffm"
    model_save_path = "./ckpt/weights.npy"

    """dataset statistic"""
    field_num = 39 # num field, f
    feature_num = 303943 # num feature, n
    embedding_num = 4 # num of embeddings, k

    """learning config"""
    learning_rate = 0.1
    batch_size = 1
    num_epoch = 1



config = Config()

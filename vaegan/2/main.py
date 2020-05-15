import tensorflow as tf

from utils import mkdir_p
from vaegan import vaegan
from utils import CelebA

flags = tf.app.flags

flags.DEFINE_integer("batch_size" , 64, "batch size")
flags.DEFINE_integer("max_iters" , 600_000, "the maxmization epoch")
flags.DEFINE_integer("latent_dim" , 64, "the dim of latent code")
flags.DEFINE_float("learn_rate_init" , 0.001, "the init of learn rate")
#Please set this num of repeat by the size of your datasets.
flags.DEFINE_integer("repeat", 10000, "the numbers of repeat for your datasets")
flags.DEFINE_string("path", "//Desktop-sa1evjv/h/small_scans-tem/96x96-tem.npy", "Training Data")
flags.DEFINE_integer("op", 0, "Training or Test")


FLAGS = flags.FLAGS
if __name__ == "__main__":

    root_log_dir = "./vaeganlogs/logs/celeba_test"
    vaegan_checkpoint_dir = "./model_vaegan/model.ckpt"
    sample_path = "./vaeganSample/sample"

    mkdir_p(root_log_dir)
    mkdir_p('./model_vaegan/')
    mkdir_p(sample_path)

    model_path = vaegan_checkpoint_dir

    batch_size = FLAGS.batch_size
    max_iters = FLAGS.max_iters
    latent_dim = FLAGS.latent_dim
    data_repeat = FLAGS.repeat

    learn_rate_init = FLAGS.learn_rate_init
    #cb_ob = CelebA(FLAGS.path)
    cb_ob = CelebA(images_path=FLAGS.path, channel=1)

    vaeGan = vaegan(batch_size= batch_size, max_iters= max_iters, repeat = data_repeat,
                      model_path= model_path, data_ob= cb_ob, latent_dim= latent_dim,
                      sample_path= sample_path , log_dir= root_log_dir , learnrate_init= learn_rate_init)
    
    vaeGan.build_model_vaegan()

    vaeGan.train()
    vaeGan.test()
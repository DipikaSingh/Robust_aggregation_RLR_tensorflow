from Variables_global import args_parser
import Main_defence_V6 as MD
import tensorflow as tf
import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


if __name__=='__main__':
    tf.keras.backend.clear_session()
    args = args_parser()
    Try_sess=MD.Main(args)
    Try_sess.print_exp_details()
    Try_sess.run()
    
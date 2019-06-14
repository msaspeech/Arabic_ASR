from tensorflow.python.keras.callbacks import Callback

from etc import settings
from utils import generate_pickle_file, file_exists, create_dir, load_pickle_data
from .plot import plot_train_loss_acc


class ModelSaver(Callback):

    def __init__(self, model_name, model_path, encoder_states, drive_instance, word_level=True, output_length=16):
        super().__init__()

        self.model_name = model_name
        self.model_path = model_path
        self.word_level = word_level
        self.output_length = output_length
        self.encoder_states = encoder_states
        self.drive_instance = drive_instance

    def on_epoch_end(self, epoch, logs=None):
        # Saving training history
        # Check if directory exists
        directory_path = settings.TRAINED_MODELS_PATH + self.model_name
        if not file_exists(directory_path):
            create_dir(directory_path)

        # Word level history
        if self.word_level:
            hist_path = settings.TRAINED_MODELS_PATH + self.model_name + "/" + self.model_name + "word.pkl"
            average_accuracy = 0
            if file_exists(hist_path):
                acc_loss_history = load_pickle_data(hist_path)
            else:
                acc_loss_history = dict()
                acc_loss_history["accuracy"] = []
                acc_loss_history["loss"] = []

                # Average accuracy
            for i in range(0, 6):
                accuracy = "decoder_dense" + str(i) + "_acc"
                average_accuracy += logs[accuracy]

            average_accuracy = float(average_accuracy) / float(6)

            acc_loss_history["accuracy"].append(average_accuracy)
            acc_loss_history["loss"].append(logs["loss"])

        # Character level history
        else:
            hist_path = settings.TRAINED_MODELS_PATH + self.model_name + "/" + self.model_name + "char.pkl"
            if file_exists(hist_path):
                acc_loss_history = load_pickle_data(hist_path)
            else:
                acc_loss_history = dict()
                acc_loss_history["accuracy"] = []
                acc_loss_history["loss"] = []

            acc_loss_history["accuracy"].append(logs["acc"])
            acc_loss_history["loss"].append(logs["loss"])

        generate_pickle_file(acc_loss_history, hist_path)
        plot_train_loss_acc(hist_path, self.word_level)

        self.model.save(self.model_path)
        model_title = self.model_name

        # Saving model

        # Saving training results

        folders_dict_id = {"architecture1": "182V0L64_Ovt4i5VEdxZkzM73woulpC40",
                           "architecture2": "1nAKBM7fJMcg7VEaFHr4A-XDIilQ5w7MP",
                           "architecture3": "1KZdD44LUEaGHoPIHI7TMJ8gx_KUmOueS",
                           "architecture4": "1mPqGG38TGcGUmw0Oz4KqP4M-R5atREaO",
                           "architecture5": "1Hy0hecoGHYjiWkZpj8snZpqTiH4YSdE6",
                           "architecture6": "1YDWIu47oGj2C_C5C2XC9-51UjUTIbelw",
                           "architecture7": "18Xiiq1CEb8bxJhRY_gjUUK-2xNjY3ixw"}

        # parent_directory_id = "0B5fJkPjHLj3Jdkw5ZnFiY0lZV1U"
        # file_list = self.drive_instance.ListFile({'q': "\'"+parent_directory_id+"\'"+" in parents  and trashed=false"}).GetList()
        # for file1 in file_list:
        #    print('title: %s, id: %s   ' % (file1['title'], file1['id']))

        # parent_directory_id = '0B5fJkPjHLj3Jdkw5ZnFiY0lZV1U'
        parent_directory_id = folders_dict_id[self.model_name]
        file_list = self.drive_instance.ListFile(
            {'q': "\'" + parent_directory_id + "\'" + " in parents  and trashed=false"}).GetList()

        try:
            for file1 in file_list:
                if file1['title'] == self.model_path:
                    file1.Delete()
        except:
            print("File not found")

        uploaded = self.drive_instance.CreateFile(
            {model_title: self.model_name, "parents": [{"kind": "drive#fileLink", "id": parent_directory_id}]})
        uploaded.SetContentFile(self.model_path)
        uploaded.Upload()

        # Save training loss and accuracy
        file_list = self.drive_instance.ListFile(
            {'q': "\'" + parent_directory_id + "\'" + " in parents  and trashed=false"}).GetList()
        try:
            for file1 in file_list:
                if file1['title'] == hist_path:
                    file1.Delete()
        except:
            print("File not found")

        uploaded = self.drive_instance.CreateFile(
            {model_title: "Train history", "parents": [{"kind": "drive#fileLink", "id": parent_directory_id}]})
        uploaded.SetContentFile(hist_path)
        uploaded.Upload()

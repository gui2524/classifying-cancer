import FileSystemManager
import cnn_model
import sys_utils
import logging


logging.basicConfig(level=logging.INFO)

source_data = 'http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz'
source_archive = source_data.split('/')[-1]
image_directory = 'images'
model_directory = 'tmp'

mode_input = input("Would you like to train (1) the model or use it for a prediction (2): ")

while int(mode_input) not in [1, 2]:
    mode_input = input("Please enter 1 (Train model) or 2 (Prediction): ")

if int(mode_input) == 1:

    clean_run = input("Destroy existing resources and train from scratch " +
                      "(WARNING: this is memory intensive and may take considerable time)? Enter Y/N: ")

    while clean_run.upper() not in ['Y', 'N']:
        clean_run = input("Please enter Y (clean run) or N (retrain existing resources): ")

    if clean_run.upper() == 'Y':

        file_manager = FileSystemManager(image_directory, model_directory)
        file_manager.clean_run()

        extract_dir = file_manager.extract_archive(source_archive)
        file_manager.remove_files_except('.png')
        file_manager.data_science_fs(category0='benign', category1='malignant')
        file_manager.organise_files(extract_dir, category_rules={'benign': 'SOB_B_.*.png', 'malignant': 'SOB_M_.*.png'})

    elif clean_run.upper() == 'N':
        pass

    else:
        sys_utils.graceful_exit()

    cnn_model.train(image_directory, model_directory)


elif int(mode_input) == 2:

    print("We will now randomly select an image from our prediction set (previously unseen by our model).")

    prediction, ground_truth = cnn_model.predict(image_directory, model_directory)

    print("Prediction: This is a %s cell.\nValidation: It was a %s cell" % (prediction, ground_truth))

else:
    sys_utils.graceful_exit()

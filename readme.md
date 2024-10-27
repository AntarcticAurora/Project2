## steps:
1. In your virtual environment, download the required packages 
2. Generate captcha with your fonts:
   - place your two primary fonts in the train_font folder
   - place additional fonts for mixing in the mix_font folder
   - run on clt: 
   ```shell
   mkdir [your_output_data_dir]
   python generate.py --font_dir train_font --[max_len(captchas will be 2-max_len length)] --count [font1_generate_number] [font2_generate_number] --output_dir [your_output_data_dir] --symbols symbols.txt --mix_dir mix_font --mix_count [number of mix font generation] --train_ratio [train/val split ratio between 0 and 1]
   ```

3. Train yolo model with your generated dataset
   - run on clt: 
   ```shell
   python train.py --model_path yolo11x.pt --data_config_path [your_output_data_dir]/data.yaml --img_size 192 -n [your_train_name] -e [epoch_number] -b [batch_size]
   ```
   - the result will be saved in runs/detect/[your_train_name] folder
   - the best performing model will be saved as weights/best.pt within that folder
   - upon completion of training, the training loss plot, confusion matrix, and other metrics will be saved in the same folder for you to verify the training progress

4. Predict the testing captcha images with the trained model, and save the csv file for submitty.
   - run on clt:
   ```shell
   python predict.py -i [path_to_your_testing_captcha_folder] -m [path_to_your_trained_model] -o [your_output_folder_path] -s symbols.txt -n result.csv --save_plot
   ```
   -  "-save_plot" is to save the visualization images of the prediction
   - your .csv file will be saved under [your_output_folder_path]

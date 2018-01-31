Put your own raw images into this folder.

After you put your images, go back to previous folder then run reshape_image_size.py

Every images will reshape to (500px x [a]) or ([a] x 500px).

([a] <= 500px)

Then, use labelImg tool on our images and create xml files.

At last, seperate xml files and our images to test and train folders. (Recommend ratio>> 2:8=test:train)

([WARNING] Must copy and paste instead of move files)

After seperating files, go back to previous folder and run xml_to_csv.py.

This will create csv files to data folder.
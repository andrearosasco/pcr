## Dataset Splits
Each directory (e.g. easy, hard, etc.) contains a different split generated through the `datasets.DatasetGenerator.py` script.
Each split consist in three text files listing the dataset samples to use during training, testing and validation.

### Easy
Contains the classes with the highest number of samples + the microwave class (for its resemblance with a box)
- Classes: airplane, cabinet, car, chair, lamp, sofa, table, vessel, microwave  
- Oversampling: True
### Hard
Contains all of the 55 ShapeNetCore classes
- Classes: all
- Oversampling: True